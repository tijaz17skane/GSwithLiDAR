# Affine vs Similarity Transformation for Camera Alignment

## Overview

This document explains the differences between the two alignment approaches in SRTaligner.py and SRTalignerAffine.py.

## Similarity Transformation (SRTaligner.py)

**What it does:**
- Uses Umeyama's method to find Scale + Rotation + Translation (SRT)
- Preserves angles and shapes (conformal transformation)
- Uniform scaling only (same scale factor for all axes)

**Mathematical form:**
```
Y = s * R * X + t
```
Where:
- `s` is a scalar (uniform scale factor)
- `R` is a 3x3 rotation matrix (orthogonal)
- `t` is a 3D translation vector

**Constraints:**
- 7 degrees of freedom (3 rotation + 1 scale + 3 translation)
- Preserves angles between vectors
- Preserves relative distances (up to uniform scaling)

## Affine Transformation (SRTalignerAffine.py)

**What it does:**
- Uses least squares to find a general 3x3 matrix + translation
- Allows non-uniform scaling, shearing, and rotation
- More flexible but may distort shapes

**Mathematical form:**
```
Y = M * X + t
```
Where:
- `M` is a general 3x3 matrix (can include scaling, rotation, shear)
- `t` is a 3D translation vector

**Constraints:**
- 12 degrees of freedom (9 matrix + 3 translation)
- Can handle non-uniform scaling (different scale factors for each axis)
- Can handle shearing transformations
- May not preserve angles or relative distances

## When to Use Each

### Use Similarity Transformation when:
- Camera poses should maintain their relative geometric relationships
- You expect only global scale, rotation, and translation differences
- You want to preserve the "shape" of the camera configuration
- Working with well-calibrated camera systems

### Use Affine Transformation when:
- There might be systematic distortions in one coordinate system
- Non-uniform scaling is present (e.g., different units on different axes)
- The camera systems have different coordinate conventions
- Maximum flexibility is needed for alignment

## Output Differences

### SRTaligner.py outputs:
- `scale_matrix.txt` - 4x4 uniform scale matrix
- `rotation_matrix.txt` - 4x4 rotation matrix
- `translation_matrix.txt` - 4x4 translation matrix
- `combined_transform.txt` - Combined T*R*S matrix

### SRTalignerAffine.py outputs:
- `affine_matrix_3x3.txt` - 3x3 general affine matrix
- `translation_vector.txt` - 3D translation vector
- `combined_transform.txt` - 4x4 homogeneous transformation matrix

## Computational Differences

### Similarity (Umeyama):
- Uses SVD for robust rotation estimation
- Analytically computes optimal scale factor
- More stable for noisy data
- Constrains solution to physically meaningful transformations

### Affine (Least Squares):
- Solves overconstrained linear system
- Can fit any linear transformation
- May be sensitive to outliers
- More flexible but potentially less robust

## Example Usage

```bash
# Similarity transformation (constrained)
python SRTaligner.py --inputA cameras_A.txt --inputB cameras_B.txt --output_dir ./results_similarity

# Affine transformation (flexible)  
python SRTalignerAffine.py --inputA cameras_A.txt --inputB cameras_B.txt --output_dir ./results_affine
```

## Recommendation

Start with similarity transformation (SRTaligner.py) as it's more robust and preserves geometric relationships. If the alignment error is still high, try affine transformation (SRTalignerAffine.py) for additional flexibility.
