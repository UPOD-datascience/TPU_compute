"""
File splitter utility for large JSONL files.

Reads all JSONL files from a folder and splits each into N separate files
to help with memory issues during processing.

Usage:
    python file_splitter.py --input_dir /path/to/jsonl/files --output_dir /path/to/output --num_splits 10
"""

import argparse
import os
from pathlib import Path


def count_lines(file_path: str) -> int:
    """Count number of lines in a file without loading it into memory."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def split_file(input_path: str, output_dir: str, num_splits: int) -> None:
    """
    Split a single JSONL file into N separate files.

    Args:
        input_path: Path to the input JSONL file
        output_dir: Directory to write split files
        num_splits: Number of files to split into
    """
    input_path = Path(input_path)
    file_stem = input_path.stem  # filename without extension

    # Count total lines
    print(f"Counting lines in {input_path.name}...")
    total_lines = count_lines(input_path)
    print(f"  Total lines: {total_lines}")

    if total_lines == 0:
        print(f"  Skipping empty file: {input_path.name}")
        return

    # Calculate lines per split
    lines_per_split = total_lines // num_splits
    remainder = total_lines % num_splits

    print(f"  Splitting into {num_splits} files (~{lines_per_split} lines each)")

    # Open input file and split
    with open(input_path, "r", encoding="utf-8") as infile:
        current_split = 0
        current_line_count = 0
        outfile = None

        # Calculate how many lines this split should have
        # Distribute remainder across first few splits
        lines_for_current_split = lines_per_split + (
            1 if current_split < remainder else 0
        )

        for line in infile:
            # Open new output file if needed
            if outfile is None:
                output_path = os.path.join(
                    output_dir, f"{file_stem}_part{current_split:04d}.jsonl"
                )
                outfile = open(output_path, "w", encoding="utf-8")

            # Write line
            outfile.write(line)
            current_line_count += 1

            # Check if current split is complete
            if current_line_count >= lines_for_current_split:
                outfile.close()
                outfile = None
                current_split += 1
                current_line_count = 0
                # Calculate lines for next split
                lines_for_current_split = lines_per_split + (
                    1 if current_split < remainder else 0
                )

        # Close any remaining open file
        if outfile is not None:
            outfile.close()

    print(f"  Done! Created {current_split} split files.")


def main():
    parser = argparse.ArgumentParser(
        description="Split large JSONL files into smaller chunks"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing JSONL files to split",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to write split files"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=10,
        help="Number of files to split each input file into (default: 10)",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JSONL files in input directory
    jsonl_files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]

    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL file(s) to split")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of splits per file: {args.num_splits}")
    print("-" * 50)

    for jsonl_file in sorted(jsonl_files):
        input_path = os.path.join(args.input_dir, jsonl_file)
        print(f"\nProcessing: {jsonl_file}")
        split_file(input_path, args.output_dir, args.num_splits)

    print("\n" + "=" * 50)
    print("All files split successfully!")


if __name__ == "__main__":
    main()
