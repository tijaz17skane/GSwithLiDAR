#!/usr/bin/env python3
"""
Test script to verify the visibility-based optimization functionality.
This script tests the get_visible_points_mask function and visibility logic.
"""

import torch
import sys
import os
import numpy as np

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import get_visible_points_mask

def test_visibility_function():
    """Test the visibility mask function"""
    
    # Create dummy camera parameters
    class DummyCamera:
        def __init__(self):
            self.R = torch.eye(3, device="cuda")  # Identity rotation
            self.t = torch.zeros(3, device="cuda")  # No translation
            self.width = 1920
            self.height = 1080
            self.fx = 1000.0
            self.fy = 1000.0
            self.cx = 960.0
            self.cy = 540.0
    
    camera = DummyCamera()
    
    # Create test points
    # Points in front of camera (should be visible)
    points_front = torch.tensor([
        [0, 0, 5],    # Center, in front
        [1, 1, 5],    # Slightly offset, in front
        [-1, -1, 5],  # Slightly offset, in front
    ], device="cuda", dtype=torch.float32)
    
    # Points behind camera (should not be visible)
    points_behind = torch.tensor([
        [0, 0, -5],   # Behind camera
    ], device="cuda", dtype=torch.float32)
    
    # Points outside frustum (should not be visible)
    points_outside = torch.tensor([
        [1000, 0, 5],  # Far to the right
        [0, 1000, 5],  # Far up
    ], device="cuda", dtype=torch.float32)
    
    # Test visibility for different point sets
    print("Testing visibility function...")
    
    # Test points in front
    visible_front = get_visible_points_mask(points_front, camera)
    print(f"Points in front: {visible_front}")
    print(f"Expected: [True, True, True]")
    print(f"Result: {visible_front.cpu().numpy()}")
    
    # Test points behind
    visible_behind = get_visible_points_mask(points_behind, camera)
    print(f"Points behind: {visible_behind}")
    print(f"Expected: [False]")
    print(f"Result: {visible_behind.cpu().numpy()}")
    
    # Test points outside frustum
    visible_outside = get_visible_points_mask(points_outside, camera)
    print(f"Points outside: {visible_outside}")
    print(f"Expected: [False, False]")
    print(f"Result: {visible_outside.cpu().numpy()}")
    
    # Test combined points
    all_points = torch.cat([points_front, points_behind, points_outside], dim=0)
    visible_all = get_visible_points_mask(all_points, camera)
    print(f"All points: {visible_all}")
    print(f"Expected: [True, True, True, False, False, False]")
    print(f"Result: {visible_all.cpu().numpy()}")
    
    print("\nVisibility test completed!")
    print("If the results match expectations, the visibility function is working correctly.")

if __name__ == "__main__":
    test_visibility_function() 