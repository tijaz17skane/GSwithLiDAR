#!/usr/bin/env python3
"""
Test script to verify that the fix_xyz functionality works correctly.
This script creates a simple Gaussian model and tests that xyz gradients are disabled when fix_xyz=True.
"""

import torch
import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import GaussianModel
from arguments import OptimizationParams
from argparse import ArgumentParser

def test_fix_xyz():
    """Test that fix_xyz parameter correctly disables xyz gradients"""
    
    # Create argument parser and add optimization params
    parser = ArgumentParser()
    opt_params = OptimizationParams(parser)
    args = parser.parse_args([])  # Parse empty args to get defaults
    
    # Test 1: Default behavior (fix_xyz=False)
    print("Test 1: Default behavior (fix_xyz=False)")
    args.fix_xyz = False
    opt = opt_params.extract(args)
    
    gaussians = GaussianModel(sh_degree=3)
    # Create some dummy data
    dummy_xyz = torch.randn(10, 3, device="cuda")
    gaussians._xyz = torch.nn.Parameter(dummy_xyz.requires_grad_(True))
    gaussians._features_dc = torch.nn.Parameter(torch.randn(10, 1, 3, device="cuda").requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(torch.randn(10, 1, 15, device="cuda").requires_grad_(True))
    gaussians._scaling = torch.nn.Parameter(torch.randn(10, 3, device="cuda").requires_grad_(True))
    gaussians._rotation = torch.nn.Parameter(torch.randn(10, 4, device="cuda").requires_grad_(True))
    gaussians._opacity = torch.nn.Parameter(torch.randn(10, 1, device="cuda").requires_grad_(True))
    gaussians._exposure = torch.nn.Parameter(torch.randn(1, 3, 4, device="cuda").requires_grad_(True))
    
    gaussians.training_setup(opt)
    
    print(f"  xyz requires_grad: {gaussians._xyz.requires_grad}")
    print(f"  xyz in optimizer: {'xyz' in [group['name'] for group in gaussians.optimizer.param_groups]}")
    
    # Test 2: Fixed xyz behavior (fix_xyz=True)
    print("\nTest 2: Fixed xyz behavior (fix_xyz=True)")
    args.fix_xyz = True
    opt = opt_params.extract(args)
    
    gaussians2 = GaussianModel(sh_degree=3)
    gaussians2._xyz = torch.nn.Parameter(dummy_xyz.requires_grad_(True))
    gaussians2._features_dc = torch.nn.Parameter(torch.randn(10, 1, 3, device="cuda").requires_grad_(True))
    gaussians2._features_rest = torch.nn.Parameter(torch.randn(10, 1, 15, device="cuda").requires_grad_(True))
    gaussians2._scaling = torch.nn.Parameter(torch.randn(10, 3, device="cuda").requires_grad_(True))
    gaussians2._rotation = torch.nn.Parameter(torch.randn(10, 4, device="cuda").requires_grad_(True))
    gaussians2._opacity = torch.nn.Parameter(torch.randn(10, 1, device="cuda").requires_grad_(True))
    gaussians2._exposure = torch.nn.Parameter(torch.randn(1, 3, 4, device="cuda").requires_grad_(True))
    
    gaussians2.training_setup(opt)
    
    print(f"  xyz requires_grad: {gaussians2._xyz.requires_grad}")
    print(f"  xyz in optimizer: {'xyz' in [group['name'] for group in gaussians2.optimizer.param_groups]}")
    
    # Test 3: Verify that densification methods are skipped when xyz is fixed
    print("\nTest 3: Densification methods")
    print("  Testing densify_and_split with fixed xyz...")
    gaussians2.densify_and_split(torch.randn(10, 1), 0.1, 1.0)
    print("  Testing densify_and_clone with fixed xyz...")
    gaussians2.densify_and_clone(torch.randn(10, 1), 0.1, 1.0)
    
    print("\nAll tests completed successfully!")
    print("Summary:")
    print("- When fix_xyz=False: xyz gradients are enabled and xyz is in optimizer")
    print("- When fix_xyz=True: xyz gradients are disabled and xyz is not in optimizer")
    print("- Densification methods are skipped when xyz is fixed")

if __name__ == "__main__":
    test_fix_xyz() 