import argparse
import os

def filter_missing_images(images_txt_path, image_folder_path, output_path):
    kept_lines = 0
    removed_lines = 0

    with open(images_txt_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            stripped = line.strip()
            if not stripped:
                continue

            tokens = stripped.split()
            if len(tokens) < 10:
                print(f"Skipping malformed line: {line.strip()}")
                continue

            image_path = tokens[-1]

            # Remove the 'images/' prefix if present
            if image_path.startswith("images/"):
                image_path = image_path[7:]

            full_path = os.path.join(image_folder_path, image_path)

            if os.path.isfile(full_path):
                outfile.write(line)
                kept_lines += 1
            else:
                print(f"Missing: {full_path} → Skipping")
                removed_lines += 1

    print(f"\n✅ Done. Kept {kept_lines} entries. Removed {removed_lines}.")
    print(f"📝 Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out missing images from images.txt")
    parser.add_argument("images_txt_path", help="Path to the input images.txt")
    parser.add_argument("image_folder_path", help="Actual folder where images are stored (e.g., .../images/images)")
    parser.add_argument("output_path", help="Where to save the cleaned images.txt")

    args = parser.parse_args()

    filter_missing_images(args.images_txt_path, args.image_folder_path, args.output_path)
