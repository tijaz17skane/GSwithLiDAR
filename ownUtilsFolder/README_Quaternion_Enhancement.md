# SRTaligner.py Enhancement: Quaternion-Aware Camera Pose Alignment

## Overview

The SRTaligner.py has been enhanced to properly handle both camera positions AND orientations during similarity transformation. Previously, it only transformed camera positions while ignoring quaternion rotations.

## Key Changes

### 1. **Robust Quaternion Handling**
- **Uses scipy.spatial.transform.Rotation** when available for robust quaternion operations
- **Graceful fallback** to manual quaternion functions when scipy is not available
- **Automatic detection** and reporting of which method is being used

### 2. **New Quaternion Functions**
```python
# Robust quaternion conversion using scipy (preferred)
quaternion_from_rotation_matrix()  # Matrix → Quaternion
quaternion_to_rotation_matrix()    # Quaternion → Matrix  
quaternion_multiply_robust()       # Quaternion multiplication

# Manual fallback functions (when scipy unavailable)
quaternion_from_rotation_matrix_manual()
quaternion_multiply()
```

### 3. **Complete Camera Pose Transformation**
- **Before**: Only transformed camera positions (TX, TY, TZ)
- **After**: Transforms both positions AND orientations (quaternions)

```python
# New transformation pipeline
def transform_camera_poses(positions, quaternions, R, t, s):
    transformed_positions = transform_points(positions, R, t, s)
    transformed_quaternions = transform_quaternions(quaternions, R)
    return transformed_positions, transformed_quaternions
```

### 4. **Enhanced Loading Functions**
```python
load_camera_poses_with_quaternions()  # Loads positions + quaternions
load_colmap_cameras_with_quaternions() # COLMAP-specific loader
```

### 5. **Improved Output Functions**
```python
save_colmap_format_with_quaternions()  # Saves aligned poses with rotations
```

## Mathematical Background

### Similarity Transformation for Camera Poses

When applying similarity transformation (Scale + Rotation + Translation):

1. **Position Transformation:**
   ```
   new_position = s * R * original_position + t
   ```

2. **Orientation Transformation:**
   ```
   new_quaternion = R_quaternion * original_quaternion
   ```

### Quaternion Formats

- **COLMAP Format**: [qw, qx, qy, qz] (w-first)
- **Scipy Format**: [qx, qy, qz, qw] (w-last)
- **Automatic conversion** between formats when using scipy

## Usage Examples

### Basic Usage (Recommended)
```bash
# Standard usage with quaternion-aware transformation
python SRTaligner.py --inputA source_cameras.txt --inputB target_cameras.txt --output_dir ./results

# With detailed parameters and verbose output
python SRTaligner.py --inputA source_cameras.txt --inputB target_cameras.txt --output_dir ./results --save_params -v
```

### Input Format Specification
```bash
# When inputs are in camera coordinate system
python SRTaligner.py --inputA cameras_A.txt --inputB cameras_B.txt --input_format cam --output_dir ./results

# When inputs are in world coordinate system  
python SRTaligner.py --inputA cameras_A.txt --inputB cameras_B.txt --input_format world --output_dir ./results
```

## Expected Input Format

```
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 32
1 -0.64361773 0.72291991 -0.20270558 0.14850409 1.26079431 2.49051570 -0.03120070 1 front_image001.jpg

2 -0.64289423 0.72379351 -0.20099820 0.14969866 1.15902908 2.13787318 -0.02584173 1 front_image002.jpg
```

## Enhanced Outputs

### 1. **Complete Pose Alignment**
- `aligned_in_cam.txt` - Aligned cameras with **both positions and orientations**
- Preserves original COLMAP structure
- Maintains camera coordinate system conventions

### 2. **Visualization Files**  
- `aligned_in_world.ply` - Aligned camera positions for 3D visualization
- `aligned_with_this.ply` - Target reference positions for comparison

### 3. **Transformation Matrices**
- `scale_matrix.txt` - 4x4 uniform scale matrix
- `rotation_matrix.txt` - 4x4 rotation matrix
- `translation_matrix.txt` - 4x4 translation matrix
- `combined_transform.txt` - Combined T*R*S matrix

### 4. **Status Reporting**
```
✓ Using scipy.spatial.transform for robust quaternion operations
✓ Transformed all 32 camera positions and orientations from source A
✓ Aligned cameras with orientations (COLMAP): aligned_in_cam.txt
```

## Benefits

### 1. **Mathematically Correct**
- Transforms both camera positions AND orientations
- Maintains geometric relationships between cameras
- Preserves camera coordinate system conventions

### 2. **Robust Implementation**
- Uses scipy when available for numerical stability
- Graceful fallback to manual functions
- Proper quaternion format handling

### 3. **Backward Compatible**
- Works with existing COLMAP workflows
- Maintains all original output formats
- Same command-line interface

### 4. **Better Alignment Quality**
- Complete 6DOF (position + orientation) transformation
- Improved alignment accuracy for camera systems
- Suitable for photogrammetry and 3D reconstruction

## Dependencies

### Required
- numpy
- argparse (built-in)
- cam_world_conversions (local module)

### Optional (Recommended)
- scipy.spatial.transform (for robust quaternion operations)

## Installation Note

If scipy is not available, the tool will automatically use manual quaternion functions with a warning. For best results, install scipy:

```bash
pip install scipy
```

This enhancement makes SRTaligner.py suitable for complete camera pose alignment in photogrammetry, 3D reconstruction, and computer vision applications where both camera positions and orientations matter.
