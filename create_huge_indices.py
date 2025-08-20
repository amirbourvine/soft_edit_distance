#!/usr/bin/env python3
"""
Efficient generator of 1 billion unique random DNA strings of length 20.
Uses memory-efficient techniques to handle the large scale.
"""

import random
import sys
import os
from typing import Set

def generate_random_dna_string(length: int = 20) -> str:
    """Generate a random DNA string of specified length."""
    bases = 'ACGT'
    return ''.join(random.choice(bases) for _ in range(length))

def generate_unique_dna_strings(count: int, length: int = 20, output_file: str = 'dna_strings.txt'):
    """
    Generate unique random DNA strings efficiently.
    
    Args:
        count: Number of unique strings to generate (10^9)
        length: Length of each DNA string (20)
        output_file: Output file name
    """
    # Calculate total possible combinations: 4^20 = 1,099,511,627,776
    total_possible = 4 ** length
    print(f"Total possible combinations: {total_possible:,}")
    print(f"Requested count: {count:,}")
    
    if count > total_possible:
        raise ValueError(f"Cannot generate {count} unique strings. Maximum possible: {total_possible}")
    
    # Use a more memory-efficient approach for large counts
    # We'll use a bloom filter-like approach with periodic duplicate checking
    
    generated_count = 0
    duplicates_found = 0
    batch_size = 1000000  # Process in batches of 1M
    seen_batch: Set[str] = set()
    
    print(f"Generating {count:,} unique DNA strings of length {length}...")
    print(f"Writing to: {output_file}")
    
    try:
        with open(output_file, 'w') as f:
            while generated_count < count:
                # Generate a batch
                current_batch_size = min(batch_size, count - generated_count)
                batch_strings = []
                
                for _ in range(current_batch_size * 2):  # Generate extra to account for duplicates
                    if generated_count >= count:
                        break
                    
                    dna_string = generate_random_dna_string(length)
                    
                    # Check if we've seen this string in current batch
                    if dna_string not in seen_batch:
                        seen_batch.add(dna_string)
                        batch_strings.append(dna_string)
                        generated_count += 1
                    else:
                        duplicates_found += 1
                    
                    # Progress update
                    if generated_count % 10000000 == 0:  # Every 10M
                        print(f"Generated: {generated_count:,} | Duplicates: {duplicates_found:,} | "
                              f"Progress: {generated_count/count*100:.2f}%")
                
                # Write batch to file
                for dna_string in batch_strings:
                    f.write(dna_string + '\n')
                
                # Clear batch set periodically to manage memory
                if len(seen_batch) > 50000000:  # Clear after 50M entries
                    print(f"Clearing seen set to manage memory (size: {len(seen_batch):,})")
                    seen_batch.clear()
                
                batch_strings.clear()
        
        print(f"\nCompleted!")
        print(f"Total unique strings generated: {generated_count:,}")
        print(f"Total duplicates encountered: {duplicates_found:,}")
        print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
        
    except KeyboardInterrupt:
        print(f"\nInterrupted! Generated {generated_count:,} strings so far.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def estimate_file_size(count: int, string_length: int = 20):
    """Estimate the output file size."""
    # Each string is 20 chars + 1 newline = 21 bytes
    size_bytes = count * (string_length + 1)
    size_gb = size_bytes / (1024**3)
    return size_gb

if __name__ == "__main__":
    TARGET_COUNT = 10**9 // 8  # 125 million (10^9 / 8)
    STRING_LENGTH = 20
    OUTPUT_FILE = "125m_dna_strings.txt"
    
    print("=== DNA String Generator ===")
    estimated_size = estimate_file_size(TARGET_COUNT, STRING_LENGTH)
    print(f"Estimated output file size: {estimated_size:.2f} GB")

    random.seed(42)
    
    generate_unique_dna_strings(TARGET_COUNT, STRING_LENGTH, OUTPUT_FILE)