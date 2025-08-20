def extract_first_million_lines(input_file, output_file, num_lines=1_000_000):
    """
    Extract the first million lines from a large file and save to a new file.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        num_lines: Number of lines to extract (default: 1 million)
    """
    print(f"Extracting first {num_lines:,} lines from {input_file}...")
    
    lines_written = 0
    
    try:
        with open(input_file, 'r') as infile:
            with open(output_file, 'w') as outfile:
                for line_num, line in enumerate(infile, 1):
                    if line_num > num_lines:
                        break
                    
                    outfile.write(line)
                    lines_written += 1
                    
                    # Progress indicator
                    if line_num % 100_000 == 0:
                        print(f"  Processed {line_num:,} lines...")
        
        print(f"Successfully extracted {lines_written:,} lines to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Extract first 1 million lines from indices_1481653.txt
    extract_first_million_lines('indices_1481653.txt', 'indices_1m.txt')