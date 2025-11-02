import argparse

def extract_name_xyz(input, output_path):
    with open(input, 'r') as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.strip() == "" or line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        name = parts[-1]
        tx, ty, tz = parts[5:8]
        out_lines.append(f"{name} {tx} {ty} {tz}\n")
    with open(output_path, 'w') as f:
        f.writelines(out_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract_name_xyz(args.input, args.output)
    print(f"Output written to {args.output}")