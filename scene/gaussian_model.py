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
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, lidar_point_mask=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # Store initial points for Mahalanobis distance loss
        self._initial_points3D = fused_point_cloud.detach().clone()
        
        # Store LiDAR point mask (which points should be used as parents for Mahalanobis loss)
        if lidar_point_mask is not None:
            self._lidar_point_mask = torch.tensor(lidar_point_mask, device="cuda", dtype=torch.bool)
            print(f"LiDAR mask loaded: {self._lidar_point_mask.sum().item()}/{len(self._lidar_point_mask)} points are LiDAR points")
        else:
            self._lidar_point_mask = None
            print("No LiDAR mask provided, using all points for Mahalanobis loss")

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
        
        # Initialize parent mapping: only LiDAR gaussians are tracked (one-time setup, no maintenance during densification)
        # parent_idx[i] = initial point index that gaussian i is tied to, or -1 if not a LiDAR gaussian
        if self._lidar_point_mask is not None:
            # Only LiDAR points get a valid parent index
            self.parent_idx = torch.where(
                self._lidar_point_mask,
                torch.arange(self.get_xyz.shape[0], device="cuda", dtype=torch.long),
                torch.tensor(-1, device="cuda", dtype=torch.long)
            )
            # Store indices of LiDAR gaussians for fast lookup
            self._lidar_gaussian_indices = torch.where(self._lidar_point_mask)[0]
            print(f"[Graph] Initialized {len(self._lidar_gaussian_indices)} LiDAR gaussians for Mahalanobis loss (one-time, no densification tracking)")
        else:
            # No LiDAR mask - set empty indices to disable Mahalanobis loss
            self.parent_idx = torch.full((self.get_xyz.shape[0],), -1, device="cuda", dtype=torch.long)
            self._lidar_gaussian_indices = torch.tensor([], device="cuda", dtype=torch.long)
            print(f"[Graph] No LiDAR mask provided - Mahalanobis loss DISABLED (would be too slow with {self.get_xyz.shape[0]} gaussians)")
        
        # Graph maintenance flag/cache (disabled - no parent maintenance during densification)
        self._graph_dirty = False
        
        # Cache for importance sampling indices
        self._cached_iis_indices = None
        self._cached_iis_target_points = None
        
        # Flag to control whether children of LiDAR gaussians are tracked during densification
        self._track_lidar_children = False  # Default: don't track children
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        # Set flag for tracking children of LiDAR gaussians during densification
        if hasattr(training_args, 'graph_maha_track_children'):
            self._track_lidar_children = training_args.graph_maha_track_children

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

    def reset_largest_volume_scales(self, percentage=0.1, reset_scale_value=None, statistic="mean"):
        """
        Reset scales of gaussians that exceed thresholds in scale components, volume, or surface area.
        
        Step 1: Filter indices of top A% largest X, Y, and Z scale gaussians (separately)
        Step 2: Filter indices of top A% largest volume gaussians
        Step 3: Filter indices of top A% largest surface area gaussians
        
        Then reset scale X, scale Y, and scale Z to median/mean of all scale X, scale Y, and scale Z
        for all gaussians selected in steps 1, 2, and 3.
        
        Args:
            percentage: Percentage of gaussians to reset for each metric (e.g., 0.1 for top 10%)
            reset_scale_value: The scale value to reset to (in log space). 
                              If None, uses the mean or median of current scales.
            statistic: "mean" or "median" - which statistic to use when reset_scale_value is None
        """
        with torch.no_grad():
            scales = self.get_scaling  # Shape: (N, 3)
            scale_x = scales[:, 0]  # Shape: (N,)
            scale_y = scales[:, 1]  # Shape: (N,)
            scale_z = scales[:, 2]  # Shape: (N,)
            
            # Calculate volumes: volume = scale_x * scale_y * scale_z
            volumes = scales.prod(dim=1)  # Shape: (N,)
            
            # Calculate surface area approximation: SA ≈ scale_x*scale_y + scale_y*scale_z + scale_x*scale_z
            surface_areas = (scale_x * scale_y + scale_y * scale_z + scale_x * scale_z)  # Shape: (N,)
            
            # Find the number of gaussians to reset for each metric
            num_gaussians = scales.shape[0]
            num_to_reset = max(1, int(num_gaussians * percentage))
            
            # Step 1: Filter indices of top A% largest X, Y, and Z scale gaussians (separately)
            _, largest_scale_x_indices = torch.topk(scale_x, num_to_reset, largest=True)
            _, largest_scale_y_indices = torch.topk(scale_y, num_to_reset, largest=True)
            _, largest_scale_z_indices = torch.topk(scale_z, num_to_reset, largest=True)
            
            # Step 2: Filter indices of top A% largest volume gaussians
            _, largest_volume_indices = torch.topk(volumes, num_to_reset, largest=True)
            
            # Step 3: Filter indices of top A% largest surface area gaussians
            _, largest_surface_area_indices = torch.topk(surface_areas, num_to_reset, largest=True)
            
            # Combine all indices (use union to avoid double-resetting)
            all_indices = torch.cat([
                largest_scale_x_indices,
                largest_scale_y_indices,
                largest_scale_z_indices,
                largest_volume_indices,
                largest_surface_area_indices
            ])
            unique_indices = torch.unique(all_indices)
            num_to_reset_actual = unique_indices.shape[0]
            
            # Determine reset scale values for each component
            if reset_scale_value is None:
                # Use mean or median of current scales as reset value for each component
                if statistic.lower() == "median":
                    reset_scale_x = torch.median(scale_x)
                    reset_scale_y = torch.median(scale_y)
                    reset_scale_z = torch.median(scale_z)
                else:  # default to mean
                    reset_scale_x = torch.mean(scale_x)
                    reset_scale_y = torch.mean(scale_y)
                    reset_scale_z = torch.mean(scale_z)
                
                # Convert to log space
                reset_scale_log = self.scaling_inverse_activation(torch.stack([reset_scale_x, reset_scale_y, reset_scale_z]))
            else:
                # Use provided reset value (should be in log space)
                if isinstance(reset_scale_value, torch.Tensor):
                    reset_scale_log = reset_scale_value
                else:
                    # Convert scalar or list to tensor
                    reset_scale_log = torch.tensor(reset_scale_value, device=scales.device, dtype=scales.dtype)
                    if reset_scale_log.dim() == 0:
                        reset_scale_log = reset_scale_log.repeat(3)  # Broadcast to (3,)
            
            # Ensure reset_scale_log is shape (3,)
            if reset_scale_log.dim() == 0:
                reset_scale_log = reset_scale_log.unsqueeze(0).repeat(3)
            elif reset_scale_log.shape[0] != 3:
                reset_scale_log = reset_scale_log[:3] if reset_scale_log.shape[0] > 3 else torch.cat([reset_scale_log, reset_scale_log[-1:].repeat(3 - reset_scale_log.shape[0])])
            
            # Create new scaling tensor with reset values
            new_scaling = self._scaling.clone()
            new_scaling[unique_indices] = reset_scale_log.unsqueeze(0).repeat(num_to_reset_actual, 1)
            
            # Update the optimizer
            optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
            self._scaling = optimizable_tensors["scaling"]
            
            print(f"[Scale Reset] Reset scales of {num_to_reset_actual} gaussians based on:")
            print(f"  - Step 1: {len(largest_scale_x_indices)} by scale_x, {len(largest_scale_y_indices)} by scale_y, {len(largest_scale_z_indices)} by scale_z")
            print(f"  - Step 2: {len(largest_volume_indices)} by volume")
            print(f"  - Step 3: {len(largest_surface_area_indices)} by surface area")
            print(f"[Scale Reset] Using {statistic} statistic for reset scale value")
            print(f"[Scale Reset] Reset scale value (log space): {reset_scale_log.cpu().numpy()}")
            print(f"[Scale Reset] Reset scale value (actual): {self.scaling_activation(reset_scale_log).cpu().numpy()}")

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
        
        # Update LiDAR gaussian indices after pruning
        if hasattr(self, '_lidar_gaussian_indices') and self._lidar_gaussian_indices is not None:
            # Create mapping from old indices to new indices
            old_to_new = torch.full((mask.shape[0],), -1, dtype=torch.long, device=mask.device)
            old_to_new[valid_points_mask] = torch.arange(valid_points_mask.sum(), device=mask.device)
            # Update LiDAR indices
            new_lidar_indices = old_to_new[self._lidar_gaussian_indices]
            # Keep only valid (not pruned) LiDAR gaussians
            valid_lidar_mask = new_lidar_indices >= 0
            self._lidar_gaussian_indices = new_lidar_indices[valid_lidar_mask]
        
        # Invalidate cached IIS indices since topology changed
        if hasattr(self, '_cached_iis_indices'):
            self._cached_iis_indices = None
            self._cached_iis_target_points = None
        
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_parent_idx, new_is_lidar_derived=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        old_num_gaussians = self.get_xyz.shape[0]
        
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
        
        # Extend LiDAR gaussian indices for new LiDAR-derived gaussians
        if hasattr(self, '_lidar_gaussian_indices') and self._lidar_gaussian_indices is not None and new_is_lidar_derived is not None:
            # new_is_lidar_derived is a boolean mask indicating which new gaussians are LiDAR-derived
            new_lidar_local_indices = torch.where(new_is_lidar_derived)[0]
            new_lidar_global_indices = new_lidar_local_indices + old_num_gaussians
            self._lidar_gaussian_indices = torch.cat((self._lidar_gaussian_indices, new_lidar_global_indices))
        
        # Invalidate cached IIS indices since topology changed
        self._cached_iis_indices = None
        self._cached_iis_target_points = None
        
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
        
        # Track which new gaussians are LiDAR-derived (inherit from parent) - only if tracking is enabled
        new_is_lidar_derived = None
        if self._track_lidar_children and hasattr(self, '_lidar_gaussian_indices') and self._lidar_gaussian_indices is not None:
            # Check if selected points are LiDAR-derived (have valid parent >= 0)
            selected_parents = self.parent_idx[selected_pts_mask]
            is_lidar = selected_parents >= 0
            new_is_lidar_derived = is_lidar.repeat(N)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, base_parent, new_is_lidar_derived)

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
        
        # Track which new gaussians are LiDAR-derived (inherit from parent) - only if tracking is enabled
        new_is_lidar_derived = None
        if self._track_lidar_children and hasattr(self, '_lidar_gaussian_indices') and self._lidar_gaussian_indices is not None:
            # Check if selected points are LiDAR-derived (have valid parent >= 0)
            selected_parents = self.parent_idx[selected_pts_mask]
            new_is_lidar_derived = selected_parents >= 0
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, base_parent, new_is_lidar_derived)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        # Skip parent maintenance - we only track original LiDAR gaussians now
        # (parent assignments are maintained via _lidar_gaussian_indices during prune/densify)
        self._graph_dirty = False
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

    def compute_graph_mahalanobis_loss(self, threshold_sigma=1e-6, gaussian_batch_size=1024, 
                                        iis_sample_indices=None, debug_timing=False):
        """
        Graph-based Mahalanobis loss (optimized for LiDAR-only tracking):
        - Only original LiDAR gaussians are tracked (no densification tracking)
        - Each LiDAR gaussian is tied to its initial point
        - Compute squared Mahalanobis distance from gaussian mean to initial point using gaussian's covariance
        - If iis_sample_indices provided, only compute for those points
        
        Args:
            threshold_sigma: Regularization for covariance diagonal
            gaussian_batch_size: Batch size for processing gaussians
            iis_sample_indices: If provided, only compute loss for these LiDAR point indices (importance sampling)
            debug_timing: If True, print timing for each section
        """
        import time
        if debug_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        
        assert hasattr(self, 'parent_idx') and hasattr(self, '_initial_points3D'), "parent_idx/_initial_points3D missing"
        if self._initial_points3D is None or self.get_xyz.shape[0] == 0:
            return torch.tensor(0.0, device="cuda")
        
        # Get LiDAR gaussian indices (only these are tracked)
        if hasattr(self, '_lidar_gaussian_indices') and self._lidar_gaussian_indices is not None:
            lidar_indices = self._lidar_gaussian_indices
        else:
            # Fallback: use all gaussians with valid parent
            lidar_indices = torch.where(self.parent_idx >= 0)[0]
        
        if len(lidar_indices) == 0:
            return torch.tensor(0.0, device="cuda")
        
        if debug_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"  [timing] get lidar_indices: {(t1-t0)*1000:.2f}ms")
        
        # Filter to only sampled indices if importance sampling is enabled
        if iis_sample_indices is not None:
            # iis_sample_indices are parent point indices, filter lidar_indices to those
            # Use GPU-based set membership check instead of Python loop
            parent_of_lidar = self.parent_idx[lidar_indices]
            # Create a boolean mask on GPU: check if each parent is in the sample set
            sample_mask = torch.zeros(self._initial_points3D.shape[0], dtype=torch.bool, device=lidar_indices.device)
            sample_mask[iis_sample_indices] = True
            mask = sample_mask[parent_of_lidar]
            lidar_indices = lidar_indices[mask]
            
            if len(lidar_indices) == 0:
                return torch.tensor(0.0, device="cuda")
        
        if debug_timing:
            torch.cuda.synchronize()
            t2 = time.time()
            print(f"  [timing] filter by IIS: {(t2-t1)*1000:.2f}ms")
        
        # Get data only for LiDAR gaussians (compute covariance only for these, not all)
        means = self.get_xyz[lidar_indices]  # (L, 3)
        # Compute covariance only for selected gaussians instead of all
        scaling_subset = self.get_scaling[lidar_indices]
        rotation_subset = self._rotation[lidar_indices]
        
        if debug_timing:
            torch.cuda.synchronize()
            t3 = time.time()
            print(f"  [timing] gather means/scaling/rotation: {(t3-t2)*1000:.2f}ms")
        
        cov6 = self.covariance_activation(scaling_subset, 1, rotation_subset)  # (L, 6)
        
        if debug_timing:
            torch.cuda.synchronize()
            t4 = time.time()
            print(f"  [timing] covariance_activation: {(t4-t3)*1000:.2f}ms")
        
        parents = self.parent_idx[lidar_indices]  # (L,)
        
        num_init = self._initial_points3D.shape[0]
        
        reg = threshold_sigma
        # Diagonal variances: cov6 = [xx, xy, xz, yy, yz, zz] → diag = [xx, yy, zz] + reg
        var_x = cov6[:, 0] + reg
        var_y = cov6[:, 3] + reg
        var_z = cov6[:, 5] + reg
        
        if debug_timing:
            torch.cuda.synchronize()
            t5 = time.time()
            print(f"  [timing] prep variances/accumulators: {(t5-t4)*1000:.2f}ms")
        
        # Compute all at once (no Python loop) - much faster for GPU
        L = means.shape[0]
        
        # Gather parent initial points for all gaussians at once
        p = self._initial_points3D[parents]  # (L, 3)
        d = means - p  # (L, 3)
        
        # Squared Mahalanobis with diagonal approximation
        md2 = (d[:, 0] ** 2) / var_x + (d[:, 1] ** 2) / var_y + (d[:, 2] ** 2) / var_z
        
        # Accumulate per parent using scatter_add
        sum_dist = torch.zeros(num_init, device=means.device, dtype=means.dtype)
        count = torch.zeros(num_init, device=means.device, dtype=means.dtype)
        sum_dist.scatter_add_(0, parents, md2)
        count.scatter_add_(0, parents, torch.ones_like(md2))
        
        if debug_timing:
            torch.cuda.synchronize()
            t6 = time.time()
            print(f"  [timing] compute + scatter ({L} gaussians): {(t6-t5)*1000:.2f}ms")
        
        # Per-point mean
        per_init_mean = torch.zeros_like(sum_dist)
        valid = count > 0
        per_init_mean[valid] = sum_dist[valid] / count[valid]
        
        # Store for importance sampling
        self._last_per_point_maha_loss = per_init_mean.detach().clone()
        self._last_per_point_maha_valid = valid.clone()
        
        # Average over valid points
        if valid.any():
            out = per_init_mean[valid].mean()
        else:
            out = torch.tensor(0.0, device="cuda")
        
        if debug_timing:
            torch.cuda.synchronize()
            t7 = time.time()
            print(f"  [timing] finalize + store: {(t7-t6)*1000:.2f}ms")
            print(f"  [timing] TOTAL: {(t7-t0)*1000:.2f}ms")
        
        return out

    def get_importance_sampling_indices(self, num_samples):
        """
        Get indices for importance sampling based on per-point Mahalanobis loss.
        Points with higher loss are more likely to be sampled.
        
        Args:
            num_samples: Number of points to sample
            
        Returns:
            Tensor of sampled indices
        """
        if not hasattr(self, '_last_per_point_maha_loss') or self._last_per_point_maha_loss is None:
            # No previous loss computed, return random sample from valid points
            if hasattr(self, '_lidar_point_mask') and self._lidar_point_mask is not None:
                valid_indices = torch.where(self._lidar_point_mask)[0]
            else:
                valid_indices = torch.arange(self._initial_points3D.shape[0], device=self._initial_points3D.device)
            
            num_samples = min(num_samples, len(valid_indices))
            perm = torch.randperm(len(valid_indices), device=valid_indices.device)[:num_samples]
            return valid_indices[perm]
        
        # Get valid points (those with children and optionally LiDAR)
        valid_mask = self._last_per_point_maha_valid
        if hasattr(self, '_lidar_point_mask') and self._lidar_point_mask is not None:
            valid_mask = valid_mask & self._lidar_point_mask
        
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=self._initial_points3D.device)
        
        num_samples = min(num_samples, len(valid_indices))
        
        # Get losses for valid points
        losses = self._last_per_point_maha_loss[valid_indices]
        
        # Convert to sampling probabilities (higher loss = higher probability)
        # Add small epsilon to avoid division by zero
        probs = losses + 1e-8
        probs = probs / probs.sum()
        
        # Sample indices based on importance weights
        sampled_local_indices = torch.multinomial(probs, num_samples, replacement=False)
        sampled_global_indices = valid_indices[sampled_local_indices]
        
        return sampled_global_indices
    
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
    
    # Coverage batch helper removed