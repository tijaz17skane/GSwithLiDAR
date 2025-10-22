import struct
import argparse

def read_colmap_bin(bin_file):
    """Read COLMAP points3D.bin file and return points data."""
    points = []
    with open(bin_file, 'rb') as f:
        try:
            num_points = struct.unpack('Q', f.read(8))[0]  # Read number of points
        except struct.error:
            print("Error: Unable to read the number of points. File may be corrupted.")
            return points

        for _ in range(num_points):
            try:
                point_id = struct.unpack('Q', f.read(8))[0]  # Point ID
                x, y, z = struct.unpack('3d', f.read(24))  # XYZ coordinates
                r, g, b = struct.unpack('3B', f.read(3))  # RGB colors
                f.read(1)  # Padding byte
                error = struct.unpack('d', f.read(8))[0]  # Error
                track_length = struct.unpack('Q', f.read(8))[0]  # Track length

                # Validate track length
                if track_length > 1e6:  # Arbitrary large value to catch errors
                    print(f"Warning: Unreasonable track length {track_length}. Skipping point.")
                    continue

                tracks = []
                for _ in range(track_length):
                    track_data = f.read(16)
                    if len(track_data) < 16:
                        print(f"Warning: Insufficient bytes for track element. Skipping remaining tracks.")
                        break
                    image_id, point2d_idx = struct.unpack('2Q', track_data)  # Track elements
                    tracks.append((image_id, point2d_idx))

                points.append((point_id, x, y, z, r, g, b, error, tracks))

            except struct.error as e:
                print(f"Error reading point data: {e}. Skipping point.")
                break

    return points

def write_colmap_txt(points, txt_file):
    """Write points data to COLMAP points3D.txt file."""
    with open(txt_file, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for point in points:
            point_id, x, y, z, r, g, b, error, tracks = point
            track_str = ' '.join(f'({image_id},{point2d_idx})' for image_id, point2d_idx in tracks)
            f.write(f'{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} {track_str}\n')

def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP points3D.bin to points3D.txt.')
    parser.add_argument('--input', '-i', required=True, help='Path to input points3D.bin file')
    parser.add_argument('--output', '-o', required=True, help='Path to output points3D.txt file')
    args = parser.parse_args()

    # Read binary file
    points = read_colmap_bin(args.input)

    # Write to text file
    write_colmap_txt(points, args.output)

    print(f'Converted {args.input} to {args.output}')

if __name__ == '__main__':
    main()