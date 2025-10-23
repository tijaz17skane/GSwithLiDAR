import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Output raw positions as a TXT file with columns x y z.')
    parser.add_argument('--input', required=True, help='Path to input cameras.json')
    parser.add_argument('--output', required=True, help='Path to output positions.txt')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        cameras = json.load(f)

    with open(args.output, 'w') as f:
        f.write('# x y z\n')
        for cam in cameras:
            pos = cam['position']
            f.write(f'{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n')

if __name__ == "__main__":
    main()
