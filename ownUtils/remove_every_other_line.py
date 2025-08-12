def skip_every_other_line(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i % 2 == 0:  # Keep even-numbered lines (0, 2, 4, ...)
                outfile.write(line)

# Example usage
input_path = "/mnt/data/tijaz/dataSets/data/section_33/sparse/0/points3D32.txt"
output_path = "/mnt/data/tijaz/dataSets/data/section_33/sparse/0/points3D64.txt"
skip_every_other_line(input_path, output_path)