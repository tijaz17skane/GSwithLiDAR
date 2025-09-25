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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._initial_points3D = None  # Initialize to None
        self.parent_idx = None  # Initialize to None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._initial_points3D, # New: save initial points
            self.parent_idx # New: save parent indices
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        initial_points3D, # New: load initial points
        parent_idx # New: load parent indices
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._initial_points3D = initial_points3D # Assign loaded initial points
        self.parent_idx = parent_idx # Assign loaded parent indices
        self._graph_dirty = True # Mark graph dirty to rebuild caches after restoration

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # Store initial points for Mahalanobis distance loss
        self._initial_points3D = fused_point_cloud.detach().clone()

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Initialize parent mapping: each initial gaussian maps to its initial point index
        self.parent_idx = torch.arange(self.get_xyz.shape[0], device="cuda", dtype=torch.long)
        # Graph maintenance flag/cache
        self._graph_dirty = False
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Store initial points for Mahalanobis distance loss
        self._initial_points3D = torch.tensor(xyz, dtype=torch.float, device="cuda").detach().clone()

        # Initialize parent mapping when loading
        self.parent_idx = torch.arange(self._xyz.shape[0], device="cuda", dtype=torch.long)
        # Graph maintenance flag/cache
        self._graph_dirty = False

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        # Propagate parent mapping through pruning
        if hasattr(self, 'parent_idx') and self.parent_idx is not None:
            self.parent_idx = self.parent_idx[valid_points_mask]
        # mark graph dirty so it gets refreshed after topology change
        self._graph_dirty = True

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_parent_idx):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Extend parent mapping
        if hasattr(self, 'parent_idx') and self.parent_idx is not None:
            self.parent_idx = torch.cat((self.parent_idx, new_parent_idx.to(self.parent_idx.device)))
        # mark graph dirty so consumers can skip recompute until next topology pass fixed it
        self._graph_dirty = True

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # Parent mapping for new points: inherit from selected points
        base_parent = self.parent_idx[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, base_parent)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # Parent mapping for clones: inherit from selected points
        base_parent = self.parent_idx[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, base_parent)

    def _prune_logic(self, min_opacity, extent, max_screen_size):
        """
        Helper function to create the pruning mask and call prune_points.
        Note: self.tmp_radii must be set before calling this.
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

    def _densify_and_prune_combined(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        self._prune_logic(min_opacity, extent, max_screen_size)

        # Update parent assignments and optional grouping only when topology changes
        print(f"[graph] densify_and_prune: starting parent maintenance; gaussians={self.get_xyz.shape[0]}")
        self.ensure_parent_assignments()
        self.rebuild_parent_groups()
        self._graph_dirty = False
        print(f"[graph] densify_and_prune: maintenance done")
        torch.cuda.empty_cache()

    def densify_only(self, max_grad, extent, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Graph maintenance is handled by densify_and_clone/split if parent_idx is enabled
        self._graph_dirty = False # Indicate that graph is not dirty since child calls clean it
        torch.cuda.empty_cache()

    def prune_only(self, min_opacity, extent, max_screen_size, radii):
        """
        Prunes Gaussians based on opacity and screen size criteria.
        """
        self.tmp_radii = radii
        self._prune_logic(min_opacity, extent, max_screen_size)
        self.tmp_radii = None # Clean up temporary radii after pruning

        # Update parent assignments and optional grouping after topology changes
        print(f"[graph] prune_only: starting parent maintenance; gaussians={self.get_xyz.shape[0]}")
        self.ensure_parent_assignments()
        self.rebuild_parent_groups()
        self._graph_dirty = False
        print(f"[graph] prune_only: maintenance done")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def ensure_parent_assignments(self, nn_batch_size: int = 2048):
        """
        Ensure every gaussian has a valid parent assignment.
        If any parent indices are invalid (<0), assign them to the nearest gaussian's parent.
        If no valid gaussians exist, assign to nearest initial point.
        Runs in small batches to be memory efficient.
        """
        if not hasattr(self, 'parent_idx') or self.parent_idx is None:
            return
        if self.parent_idx.dtype != torch.long:
            self.parent_idx = self.parent_idx.long()

        invalid = self.parent_idx < 0
        if not torch.any(invalid):
            print("[graph] ensure_parent_assignments: no invalid parents")
            return

        means = self.get_xyz
        device = means.device
        valid_idx = torch.nonzero(~invalid, as_tuple=False).squeeze(-1)
        missing_idx = torch.nonzero(invalid, as_tuple=False).squeeze(-1)
        print(f"[graph] ensure_parent_assignments: invalid={missing_idx.numel()} valid={valid_idx.numel()} batch={nn_batch_size}")
        if valid_idx.numel() > 0:
            src_means = means[valid_idx].detach()
            src_parents = self.parent_idx[valid_idx]
            # Assign each missing gaussian to nearest existing gaussian's parent
            for i0 in range(0, missing_idx.numel(), nn_batch_size):
                i1 = min(i0 + nn_batch_size, missing_idx.numel())
                mids = missing_idx[i0:i1]
                q = means[mids].detach()
                d = torch.cdist(q, src_means)
                nn_j = torch.argmin(d, dim=1)
                self.parent_idx[mids] = src_parents[nn_j]
                del q, d, nn_j
                if (i0 // nn_batch_size) % 8 == 0:
                    torch.cuda.empty_cache()
        else:
            # No valid gaussians yet; assign to nearest initial point
            init_pts = self._initial_points3D
            for i0 in range(0, missing_idx.numel(), nn_batch_size):
                i1 = min(i0 + nn_batch_size, missing_idx.numel())
                mids = missing_idx[i0:i1]
                q = means[mids].detach()
                d = torch.cdist(q, init_pts)
                nn_j = torch.argmin(d, dim=1)
                self.parent_idx[mids] = nn_j
                del q, d, nn_j
                if (i0 // nn_batch_size) % 8 == 0:
                    torch.cuda.empty_cache()
            print("[graph] ensure_parent_assignments: filled via nearest initial point")

    def rebuild_parent_groups(self):
        """
        Optional lightweight grouping cache. Keeps sorted child indices per parent
        using argsort and bincount to enable faster per-parent passes if needed.
        """
        if not hasattr(self, 'parent_idx') or self.parent_idx is None:
            self._sorted_child_idx = None
            self._parent_idx_sorted = None
            self._parent_offsets = None
            return
        parents = self.parent_idx
        if parents.numel() == 0:
            self._sorted_child_idx = None
            self._parent_idx_sorted = None
            self._parent_offsets = None
            return
        sorted_idx = torch.argsort(parents)
        self._sorted_child_idx = sorted_idx
        self._parent_idx_sorted = parents[sorted_idx]
        counts = torch.bincount(parents, minlength=self._initial_points3D.shape[0])
        self._parent_offsets = torch.cumsum(counts, dim=0)


    # Coverage loss removed as requested

    def compute_graph_mahalanobis_loss(self, threshold_sigma=1e-6, gaussian_batch_size=1024):
        """
        Graph-based Mahalanobis loss:
        - Each gaussian carries a parent initial point index in self.parent_idx
        - For each initial point i, gather its children gaussians G_i
        - Compute squared Mahalanobis distance from each child gaussian mean to initial point i using that gaussian's covariance (diagonal approximation)
        - Average per-initial-point over its children; if no children, contribute 0
        - Return mean over all initial points
        Memory efficient via gaussian batching.
        """
        assert hasattr(self, 'parent_idx') and hasattr(self, '_initial_points3D'), "parent_idx/_initial_points3D missing"
        if self._initial_points3D is None or self.get_xyz.shape[0] == 0:
            return torch.tensor(0.0, device="cuda")

        means = self.get_xyz  # (G,3)
        cov6 = self.get_covariance()  # (G,6)
        parents = self.parent_idx  # (G,)

        num_init = self._initial_points3D.shape[0]
        # Accumulators per initial point
        sum_dist = torch.zeros(num_init, device=means.device, dtype=means.dtype)
        count = torch.zeros(num_init, device=means.device, dtype=means.dtype)

        reg = threshold_sigma
        # precompute diagonal variances per gaussian
        # cov6: [xx, xy, xz, yy, yz, zz] → diag = [xx, yy, zz] + reg
        var_x = cov6[:, 0] + reg
        var_y = cov6[:, 3] + reg
        var_z = cov6[:, 5] + reg

        G = means.shape[0]
        bs = min(gaussian_batch_size, G)
        for g0 in range(0, G, bs):
            g1 = min(g0 + bs, G)
            m = means[g0:g1]               # (gs,3)
            pid = parents[g0:g1]           # (gs,)
            vx = var_x[g0:g1]
            vy = var_y[g0:g1]
            vz = var_z[g0:g1]

            # Filter out-of-bounds parent indices
            valid_mask = (pid >= 0) & (pid < self._initial_points3D.shape[0])
            if not valid_mask.any():
                continue
            m = m[valid_mask]
            pid = pid[valid_mask]
            vx = vx[valid_mask]
            vy = vy[valid_mask]
            vz = vz[valid_mask]

            # gather parent initial points
            p = self._initial_points3D[pid]  # (valid,3)
            d = m - p  # (valid,3)
            # squared mahalanobis with diagonal approx
            md2 = (d[:, 0] * d[:, 0]) / vx + (d[:, 1] * d[:, 1]) / vy + (d[:, 2] * d[:, 2]) / vz

            # accumulate per parent with scatter_add
            sum_dist.scatter_add_(0, pid, md2)
            count.scatter_add_(0, pid, torch.ones_like(md2))

        # Per-initial mean; zero for entries with count==0
        per_init_mean = torch.zeros_like(sum_dist)
        valid = count > 0
        per_init_mean[valid] = sum_dist[valid] / count[valid]
        # Mean over all initial points
        out = per_init_mean.mean()
        #print(f"[graph] graph_maha: done, valid={int(valid.sum().item())}/{num_init}, mean={float(out.item()):.6f}")
        return out
    
    def _icp_align(self, source_points, target_points, max_iterations=10, tolerance=1e-6, batch_size=1000):
        """
        Memory-efficient ICP alignment between source and target point clouds.
        
        Args:
            source_points: Source points (N, 3)
            target_points: Target points (M, 3)
            max_iterations: Maximum ICP iterations
            tolerance: Convergence tolerance
            batch_size: Batch size for nearest neighbor search
            
        Returns:
            Aligned source points (N, 3)
        """
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            return source_points
        
        # Start with identity transformation
        R = torch.eye(3, device=source_points.device, dtype=source_points.dtype)
        t = torch.zeros(3, device=source_points.device, dtype=source_points.dtype)
        
        aligned_points = source_points.clone()
        
        for iteration in range(max_iterations):
            # Find nearest neighbors in batches to avoid memory issues
            nearest_targets = self._find_nearest_neighbors_batched(aligned_points, target_points, batch_size)
            
            # Compute centroids
            source_centroid = torch.mean(aligned_points, dim=0)
            target_centroid = torch.mean(nearest_targets, dim=0)
            
            # Center the points
            source_centered = aligned_points - source_centroid
            target_centered = nearest_targets - target_centroid
            
            # Compute rotation using SVD
            H = source_centered.T @ target_centered  # (3, 3)
            U, _, Vt = torch.linalg.svd(H)
            R_new = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if torch.det(R_new) < 0:
                Vt[-1, :] *= -1
                R_new = Vt.T @ U.T
            
            # Compute translation
            t_new = target_centroid - R_new @ source_centroid
            
            # Apply transformation
            aligned_points = (R_new @ aligned_points.T).T + t_new
            
            # Check convergence
            if torch.norm(R_new - R) < tolerance and torch.norm(t_new - t) < tolerance:
                break
            
            R, t = R_new, t_new
        
        return aligned_points
    
    def _find_nearest_neighbors_batched(self, source_points, target_points, batch_size=1000):
        """
        Find nearest neighbors in batches to avoid memory issues.
        
        Args:
            source_points: Source points (N, 3)
            target_points: Target points (M, 3)
            batch_size: Batch size for processing
            
        Returns:
            Nearest target points for each source point (N, 3)
        """
        N = source_points.shape[0]
        nearest_targets = torch.zeros_like(source_points)
        
        # Process source points in batches
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_source = source_points[i:end_idx]  # (batch_size, 3)
            
            # Compute distances for this batch
            distances = torch.cdist(batch_source, target_points)  # (batch_size, M)
            _, nearest_indices = torch.min(distances, dim=1)  # (batch_size,)
            batch_nearest = target_points[nearest_indices]  # (batch_size, 3)
            
            nearest_targets[i:end_idx] = batch_nearest
        
        return nearest_targets
  