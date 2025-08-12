#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def get_visible_points_mask(points_3d, camera, margin=0.1):
    """
    Get visibility mask for points based on camera frustum.
    
    Args:
        points_3d: (N, 3) tensor of 3D points
        camera: Camera object with projection matrix
        margin: Margin around frustum (fraction of image size)
    
    Returns:
        visible_mask: (N,) boolean tensor indicating visible points
    """
    # Get camera parameters
    R = torch.tensor(camera.R, device=points_3d.device, dtype=points_3d.dtype)
    T = torch.tensor(camera.T, device=points_3d.device, dtype=points_3d.dtype)
    width = camera.image_width
    height = camera.image_height
    FoVx = camera.FoVx
    FoVy = camera.FoVy
    
    # Calculate focal lengths from FoV
    fx = width / (2 * np.tan(FoVx / 2))
    fy = height / (2 * np.tan(FoVy / 2))
    cx = width / 2
    cy = height / 2
    
    # Transform points to camera coordinates
    points_cam = torch.matmul(R, points_3d.T) + T.unsqueeze(1)  # (3, N)
    
    # Check if points are in front of camera (positive Z)
    in_front = points_cam[2, :] > 0
    
    # Project to image coordinates
    x = points_cam[0, :] / points_cam[2, :] * fx + cx
    y = points_cam[1, :] / points_cam[2, :] * fy + cy
    
    # Check if points are within image bounds (with margin)
    margin_x = width * margin
    margin_y = height * margin
    in_bounds = (x >= -margin_x) & (x <= width + margin_x) & (y >= -margin_y) & (y <= height + margin_y)
    
    # Combine conditions
    visible = in_front & in_bounds
    
    return visible

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Get visibility mask for optimization
        optimize_mask = None
        if hasattr(opt, 'optimize_visible_only') and opt.optimize_visible_only:
            try:
                # Get camera frustum visibility
                camera_frustum_visible = get_visible_points_mask(gaussians.get_xyz, viewpoint_cam)
                
                if hasattr(opt, 'optimize_significant_only') and opt.optimize_significant_only:
                    # We'll determine significant points after rendering
                    optimize_mask = camera_frustum_visible
                else:
                    # Optimize all visible points
                    optimize_mask = camera_frustum_visible
                    
                if optimize_mask is not None:
                    # Temporarily disable gradients for non-visible points
                    try:
                        gaussians._xyz.requires_grad_(True)  # Ensure gradients are enabled
                        # Ensure optimize_mask has the right shape for broadcasting
                        if optimize_mask.shape[0] == gaussians._xyz.shape[0]:
                            gaussians._xyz.register_hook(lambda grad: grad * optimize_mask.float().unsqueeze(1) if grad is not None else None)
                        else:
                            print(f"Warning: Shape mismatch in gradient masking - xyz: {gaussians._xyz.shape}, mask: {optimize_mask.shape}")
                    except Exception as e:
                        print(f"Warning: Error in gradient masking: {e}. Continuing without visibility filtering.")
                        optimize_mask = None
            except Exception as e:
                print(f"Warning: Visibility optimization failed: {e}. Continuing without visibility filtering.")
                if hasattr(opt, 'disable_visibility_on_error') and opt.disable_visibility_on_error:
                    print("Disabling visibility optimization for this training run.")
                    opt.optimize_visible_only = False
                optimize_mask = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        # Determine significant points if needed
        if hasattr(opt, 'optimize_significant_only') and opt.optimize_significant_only and optimize_mask is not None:
            # Use opacity and contribution to loss to determine significant points
            opacity_threshold = 0.1
            significant_opacity = gaussians.get_opacity.squeeze() > opacity_threshold
            
            # Points that contribute significantly to the loss (have high gradients)
            if viewspace_point_tensor.grad is not None:
                grad_magnitude = torch.norm(viewspace_point_tensor.grad, dim=1)
                grad_threshold = grad_magnitude.mean() + grad_magnitude.std()
                significant_grad = grad_magnitude > grad_threshold
            else:
                significant_grad = torch.ones_like(significant_opacity)
            
            # Combine conditions
            significant_mask = significant_opacity & significant_grad & optimize_mask
            optimize_mask = significant_mask

        # Apply gradient clipping to xyz if specified
        if hasattr(opt, 'xyz_grad_clip') and opt.xyz_grad_clip > 0.0 and gaussians._xyz.requires_grad:
            torch.nn.utils.clip_grad_norm_(gaussians._xyz, opt.xyz_grad_clip)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (skip if xyz is fixed)
            if iteration < opt.densify_until_iter and gaussians._xyz.requires_grad:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                # Use visibility mask for densification if enabled
                densify_filter = visibility_filter
                if hasattr(opt, 'optimize_visible_only') and opt.optimize_visible_only and optimize_mask is not None:
                    # Only densify visible points
                    try:
                        # Ensure tensors have compatible shapes
                        if visibility_filter.shape == optimize_mask.shape:
                            densify_filter = visibility_filter & optimize_mask
                        elif visibility_filter.shape[0] == optimize_mask.shape[0]:
                            # Reshape optimize_mask to match visibility_filter
                            if len(visibility_filter.shape) == 2 and len(optimize_mask.shape) == 1:
                                optimize_mask_reshaped = optimize_mask.unsqueeze(1)
                                densify_filter = visibility_filter & optimize_mask_reshaped
                            else:
                                print(f"Warning: Shape mismatch - visibility_filter: {visibility_filter.shape}, optimize_mask: {optimize_mask.shape}")
                                densify_filter = visibility_filter
                        elif visibility_filter.shape[0] == 0:
                            # visibility_filter is empty, skip visibility-based densification
                            print(f"Warning: Empty visibility_filter, skipping visibility-based densification")
                            densify_filter = visibility_filter  # Keep it empty
                        else:
                            print(f"Warning: Shape mismatch - visibility_filter: {visibility_filter.shape}, optimize_mask: {optimize_mask.shape}")
                            densify_filter = visibility_filter
                    except Exception as e:
                        print(f"Warning: Error in visibility filtering: {e}. Using original visibility filter.")
                        densify_filter = visibility_filter
                
                gaussians.add_densification_stats(viewspace_point_tensor, densify_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif iteration < opt.densify_until_iter and not gaussians._xyz.requires_grad and iteration % opt.densification_interval == 0:
                print(f"[ITER {iteration}] Skipping densification because xyz positions are fixed")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
                # Clean up gradient hooks if visibility optimization was used
                if hasattr(opt, 'optimize_visible_only') and opt.optimize_visible_only and optimize_mask is not None:
                    # Remove any gradient hooks that were added
                    if hasattr(gaussians._xyz, '_backward_hooks') and gaussians._xyz._backward_hooks is not None:
                        for hook_id in list(gaussians._xyz._backward_hooks.keys()):
                            gaussians._xyz._backward_hooks.pop(hook_id)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")