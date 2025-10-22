import argparse
import re

def round_numbers_in_line(line):
    """
    Round all numerical values in a line to 4 decimal places.
    """
    # Split the line into tokens
    tokens = line.split()
    rounded_tokens = []
    for token in tokens:
        try:
            # Try to convert to float
            num = float(token)
            # Round to 4 decimal places
            rounded = f"{num:.4f}"
            # Remove trailing zeros and dot if integer
            if '.' in rounded:
                rounded = rounded.rstrip('0').rstrip('.')
            rounded_tokens.append(rounded)
        except ValueError:
            # Not a number, keep as is
            rounded_tokens.append(token)
    return ' '.join(rounded_tokens)

def main():
    parser = argparse.ArgumentParser(description='Round numerical values in txt file to 4 decimal places.')
    parser.add_argument('--input', required=True, help='Path to input txt file')
    parser.add_argument('--output', required=True, help='Path to output txt file')
    
    args = parser.parse_args()
    
    with open(args.input, 'r') as f_in, open(args.output, 'w') as f_out:
        for line in f_in:
            rounded_line = round_numbers_in_line(line.rstrip('\n'))
            f_out.write(rounded_line + '\n')
    
    print(f"Processed {args.input} and saved to {args.output}")

if __name__ == "__main__":
    main()