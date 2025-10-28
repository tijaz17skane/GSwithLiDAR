import argparse
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract sensor positions and orientations from meta.json to TXT table.")
    parser.add_argument('--input', required=True, help='Path to meta.json')
    parser.add_argument('--output', required=True, help='Output TXT file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        meta = json.load(f)

    rows = []
    # Handle both images and spherical_images
    for key in ['images', 'spherical_images']:
        if key in meta:
            for img in meta[key]:
                if 'pose' not in img:
                    continue
                t = img['pose'].get('translation', [None, None, None])
                q = img['pose'].get('orientation_xyzw', [None, None, None, None])
                rpy = img['pose'].get('orientation_roll_pitch_yaw', [None, None, None])
                row = {
                    'sensor_id': img.get('sensor_id', ''),
                    'path': img.get('path', ''),
                    'x': t[0], 'y': t[1], 'z': t[2],
                    'qw': q[3], 'qx': q[0], 'qy': q[1], 'qz': q[2],
                    'roll': rpy[0], 'pitch': rpy[1], 'yaw': rpy[2]
                }
                rows.append(row)
    # Assign serial numbers to sensor_ids
    sensor_serials = {'front': 0, 'left': 1, 'retro': 2, 'right': 3, 'ladybug_front': 4, 'ladybug_back': 5, 'CPS': 6}
    for row in rows:
        row['serial'] = sensor_serials.get(row['sensor_id'], -1)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, sep='\t', index=False)
    print(f"Saved {len(df)} sensor poses to {args.output}")

if __name__ == "__main__":
    main()
