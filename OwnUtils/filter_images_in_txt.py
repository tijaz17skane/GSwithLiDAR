import os
import shutil
import argparse

def move_unlisted_images(images_txt, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Extract allowed filenames from images.txt
    with open(images_txt, 'r') as f:
        lines = f.readlines()

    keep_filenames = set()
    for line in lines:
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            continue
        if line[0].isdigit():
            parts = line.split()
            if len(parts) >= 10:
                image_path = parts[9]  # Full relative path
                filename = os.path.basename(image_path)
                keep_filenames.add(filename)

    # Step 2: Go through files in input folder
    moved = 0
    for fname in os.listdir(input_folder):
        fpath = os.path.join(input_folder, fname)
        if os.path.isfile(fpath) and fname not in keep_filenames:
            shutil.move(fpath, os.path.join(output_folder, fname))
            moved += 1

    print(f"\n✅ Done: Moved {moved} images to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Move images NOT listed in images.txt from input folder to output folder.")
    parser.add_argument('--txt', required=True, help='Path to images.txt')
    parser.add_argument('--input_folder', required=True, help='Folder with original images')
    parser.add_argument('--output_folder', required=True, help='Folder to move unlisted images into')
    args = parser.parse_args()

    move_unlisted_images(args.txt, args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
