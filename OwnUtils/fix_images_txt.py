#!/usr/bin/env python3
"""
Fix images.txt file by adding missing POINTS2D lines
"""

import argparse
import os
import sys

def fix_images_txt(input_path, output_path):
    """
    Fix images.txt by adding empty POINTS2D lines where missing
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if line.startswith('#') or not line:
            fixed_lines.append(lines[i])
            i += 1
            continue
        
        # This should be an image line (first line of image pair)
        parts = line.split()
        if len(parts) >= 10:  # Should have IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            fixed_lines.append(lines[i])
            
            # Check if next line exists and is a POINTS2D line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith('#'):
                    # Next line exists and is not a comment, check if it's a POINTS2D line
                    next_parts = next_line.split()
                    if len(next_parts) % 3 == 0:  # POINTS2D line has groups of 3 (X, Y, POINT3D_ID)
                        fixed_lines.append(lines[i + 1])
                        i += 2
                    else:
                        # Next line is not a valid POINTS2D line, add empty one
                        fixed_lines.append('\n')
                        i += 1
                else:
                    # No next line or it's a comment, add empty POINTS2D line
                    fixed_lines.append('\n')
                    i += 1
            else:
                # No next line, add empty POINTS2D line
                fixed_lines.append('\n')
                i += 1
        else:
            # This line doesn't look like an image line, keep it as is
            fixed_lines.append(lines[i])
            i += 1
    
    # Write fixed file
    with open(output_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed images.txt: {input_path} -> {output_path}")
    print(f"Added missing POINTS2D lines")

def main():
    parser = argparse.ArgumentParser(description='Fix images.txt by adding missing POINTS2D lines')
    parser.add_argument('--input', required=True, help='Input images.txt file')
    parser.add_argument('--output', required=True, help='Output fixed images.txt file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    fix_images_txt(args.input, args.output)

if __name__ == "__main__":
    main() 