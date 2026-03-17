#!/usr/bin/env python3
"""
Script to load a parquet file in streaming mode and write it out to partitioned parquet files.
"""

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load a parquet file in streaming mode and write to partitioned parquet files."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input parquet file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for partitioned parquet files",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10000,
        help="Number of rows per partition (default: 10000)",
    )
    return parser.parse_args()


def partition_parquet(input_path: str, output_path: str, batch_size: int):
    """
    Read a parquet file in streaming mode and write partitioned parquet files.

    Args:
        input_path: Path to the input parquet file
        output_path: Path to the output directory
        batch_size: Number of rows per output partition
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open parquet file for streaming read
    parquet_file = pq.ParquetFile(input_path)

    partition_num = 0
    buffer = None

    # Iterate through batches from the parquet file
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert batch to table
        table = pa.Table.from_batches([batch])

        if buffer is not None:
            # Concatenate with existing buffer
            buffer = pa.concat_tables([buffer, table])
        else:
            buffer = table

        # Write out complete partitions
        while buffer.num_rows >= batch_size:
            # Slice the batch_size rows
            partition_table = buffer.slice(0, batch_size)
            buffer = buffer.slice(batch_size)

            # Write partition to file
            output_file = output_dir / f"partition_{partition_num:06d}.parquet"
            pq.write_table(partition_table, output_file)
            print(
                f"Written partition {partition_num} with {partition_table.num_rows} rows to {output_file}"
            )
            partition_num += 1

    # Write any remaining rows
    if buffer is not None and buffer.num_rows > 0:
        output_file = output_dir / f"partition_{partition_num:06d}.parquet"
        pq.write_table(buffer, output_file)
        print(
            f"Written final partition {partition_num} with {buffer.num_rows} rows to {output_file}"
        )
        partition_num += 1

    print(f"\nCompleted! Created {partition_num} partitions in {output_path}")


def main():
    args = parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)

    partition_parquet(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
