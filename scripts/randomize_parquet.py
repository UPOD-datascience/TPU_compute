import glob
import os
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def merge_parquet_files(input_paths, output_path, schema=None, batch_size=50000):
    """
    Merge multiple Parquet files into a single file using incremental out-of-core processing.

    This function processes files in batches to minimize memory usage, making it
    suitable for merging large Parquet files that don't fit in memory.

    Parameters:
    -----------
    input_paths : list of str
        List of paths to input Parquet files
    output_path : str
        Path for the merged output Parquet file
    schema : pyarrow.Schema, optional
        Schema to use for the merged file. If None, uses schema from first file.
    batch_size : int
        Number of rows to process in each batch (default: 50000)

    Returns:
    --------
    str : Path to the merged file
    """
    if len(input_paths) == 0:
        raise ValueError("No input files provided for merging")

    if len(input_paths) == 1:
        print(f"Only one input file provided, copying to {output_path}")
        shutil.copy2(input_paths[0], output_path)
        return output_path

    print(
        f"Merging {len(input_paths)} Parquet files (out-of-core, batch_size={batch_size})..."
    )

    # Get schema from first file if not provided
    if schema is None:
        first_file = pq.ParquetFile(input_paths[0])
        schema = first_file.schema_arrow

    # Calculate total rows for progress bar
    total_rows = 0
    file_row_counts = []
    for path in input_paths:
        pf = pq.ParquetFile(path)
        row_count = pf.metadata.num_rows
        file_row_counts.append(row_count)
        total_rows += row_count

    print(f"Total rows to merge: {total_rows}")
    total_batches = (total_rows + batch_size - 1) // batch_size

    # Write merged file incrementally
    with pq.ParquetWriter(output_path, schema, compression="snappy") as writer:
        with tqdm(total=total_batches, desc="Merging batches") as pbar:
            for file_idx, path in enumerate(input_paths):
                try:
                    parquet_file = pq.ParquetFile(path)
                    num_batches_in_file = (
                        file_row_counts[file_idx] + batch_size - 1
                    ) // batch_size

                    # Process file in batches
                    for batch in parquet_file.iter_batches(batch_size=batch_size):
                        # Convert batch to table and cast to target schema
                        table = pa.Table.from_batches([batch])
                        table = table.cast(schema)
                        writer.write_table(table)
                        del table
                        del batch
                        pbar.update(1)

                except Exception as e:
                    print(f"Error processing file {path}: {e}")
                    raise

    print(f"Merged file saved to: {output_path}")
    return output_path


def get_parquet_files_from_dir(directory, pattern="*.parquet"):
    """
    Get all Parquet files from a directory.

    Parameters:
    -----------
    directory : str
        Path to directory containing Parquet files
    pattern : str
        Glob pattern to match files (default: "*.parquet")

    Returns:
    --------
    list of str : Sorted list of paths to Parquet files
    """
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    files.sort()  # Sort for reproducibility
    return files


def recursive_shuffle_parquet(
    input_path,
    output_path,
    window_size=50000,
    temp_dir=None,
    seed=None,
    iterations=1,
    write_every_iteration=False,
):
    """
    Shuffles a Parquet file using a batch-based approach with minimal disk usage.

    Process:
    1. Read batches from input file and shuffle each batch internally
    2. Write each shuffled batch to a temporary file
    3. Immediately after writing, add the batch file path to a list
    4. Shuffle the list of batch files
    5. Read each batch file in shuffled order, append to output, and immediately delete
    6. Repeat the process for specified number of iterations

    Parameters:
    -----------
    input_path : str
        Path to the input Parquet file
    output_path : str
        Path for the output shuffled Parquet file
    window_size : int
        Number of rows to process in each batch
    temp_dir : str, optional
        Directory to store temporary files. If None, a system temp directory is used.
    seed : int, optional
        Random seed for reproducibility
    iterations : int
        Number of times to repeat the shuffling process
    write_every_iteration : bool
        Whether to save intermediate results after each iteration
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create a temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)

    try:
        print("Reading Parquet file metadata...")
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows

        # Get original schema
        original_schema = parquet_file.schema_arrow

        print(f"Total rows in the file: {total_rows}")
        print(f"Using temporary directory: {temp_dir}")

        # Use the input path for first iteration, then use output_path for subsequent iterations
        current_input = input_path

        for iteration in range(iterations):
            print(f"Starting shuffle iteration {iteration + 1}/{iterations}")

            # Determine the current schema
            if iteration == 0:
                current_schema = original_schema
                current_file = parquet_file
            else:
                current_file = pq.ParquetFile(current_input)
                current_schema = current_file.schema_arrow

            # Create a temporary output file for this iteration
            iter_output = os.path.join(temp_dir, f"iter_{iteration}_output.parquet")

            # Step 1 & 2: Create batches and write to temp files
            print("Step 1: Creating and shuffling batches...")

            # Keep track of batch files
            batch_files = []

            # Process each batch from the input file
            for i, batch in enumerate(
                tqdm(
                    current_file.iter_batches(batch_size=window_size),
                    desc="Processing batches",
                    total=(total_rows + window_size - 1) // window_size,
                )
            ):
                # Convert to pandas, shuffle, and convert back
                df = batch.to_pandas()
                df = df.sample(frac=1.0)

                # Create a temporary file for this batch
                batch_file = os.path.join(
                    temp_dir, f"batch_{i}_{random.randint(1000000, 9999999)}.parquet"
                )

                # Use compression to save disk space
                table = pa.Table.from_pandas(df, schema=current_schema)
                pq.write_table(table, batch_file, compression="snappy")

                # Add to our list of batch files
                batch_files.append(batch_file)

                # Clean up memory
                del df
                del table

            # Step 3: Shuffle the order of batch files
            print(f"Step 2: Shuffling order of {len(batch_files)} batch files...")
            random.shuffle(batch_files)

            # Step 4: Recombine batches in shuffled order with immediate cleanup
            print("Step 3: Recombining batches and cleaning up...")

            # Create a writer for the output
            with pq.ParquetWriter(
                iter_output, current_schema, compression="snappy"
            ) as writer:
                for batch_file in tqdm(batch_files, desc="Combining batches"):
                    try:
                        # Read this batch, write to output, and immediately delete
                        table = pq.read_table(batch_file)
                        table = table.cast(
                            current_schema
                        )  # Ensure schema compatibility
                        writer.write_table(table)

                        # Clean up memory and disk
                        del table
                        os.remove(batch_file)
                    except Exception as e:
                        print(f"Warning: Error processing batch file {batch_file}: {e}")

            # Update for next iteration
            if iteration < iterations - 1:
                # Clean up previous iteration's output file if it exists
                if iteration > 0 and current_input != input_path:
                    try:
                        os.remove(current_input)
                    except Exception as e:
                        print(f"Warning: Could not remove previous iteration file: {e}")

                current_input = iter_output
                _output_path = f"{output_path.rstrip('.parquet')}_{iteration}.parquet"
                if write_every_iteration:
                    shutil.copy2(iter_output, _output_path)
                    print(f"Shuffled parquet file saved to: {_output_path}")
            else:
                # Final iteration - copy to output_path
                shutil.copy2(iter_output, output_path)
                # Clean up the final iteration file
                try:
                    os.remove(iter_output)
                except Exception as e:
                    print(f"Warning: Could not remove final iteration file: {e}")
                # Clean up previous iteration's output file if it exists (for iterations > 1)
                if iteration > 0 and current_input != input_path:
                    try:
                        os.remove(current_input)
                    except Exception as e:
                        print(f"Warning: Could not remove previous iteration file: {e}")
        print(f"Shuffled parquet file saved to: {output_path}")

    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")
        print("Done!")


def merge_and_shuffle_parquet(
    input_paths=None,
    input_dir=None,
    output_path=None,
    window_size=50000,
    temp_dir=None,
    seed=None,
    iterations=1,
    write_every_iteration=False,
    merge_only=False,
    shuffle_only=False,
    input_path=None,
):
    """
    Merge multiple Parquet files and then shuffle the result.

    Parameters:
    -----------
    input_paths : list of str, optional
        List of paths to input Parquet files to merge
    input_dir : str, optional
        Directory containing Parquet files to merge
    output_path : str
        Path for the output file
    window_size : int
        Number of rows to process in each batch during shuffling
    temp_dir : str, optional
        Directory to store temporary files
    seed : int, optional
        Random seed for reproducibility
    iterations : int
        Number of times to repeat the shuffling process
    write_every_iteration : bool
        Whether to save intermediate results after each iteration
    merge_only : bool
        If True, only merge files without shuffling
    shuffle_only : bool
        If True, skip merging (use single input_path)
    input_path : str, optional
        Single input path (for backward compatibility / shuffle_only mode)
    """
    # Determine input files
    files_to_merge = []

    if shuffle_only and input_path:
        # Single file mode - skip merging
        files_to_merge = [input_path]
    elif input_paths:
        files_to_merge = input_paths
    elif input_dir:
        files_to_merge = get_parquet_files_from_dir(input_dir)
        if not files_to_merge:
            raise ValueError(f"No Parquet files found in directory: {input_dir}")
        print(f"Found {len(files_to_merge)} Parquet files in {input_dir}")
    elif input_path:
        files_to_merge = [input_path]
    else:
        raise ValueError("Must provide either input_paths, input_dir, or input_path")

    # Create temp directory
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    else:
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False

    try:
        # Determine if we need to merge
        if len(files_to_merge) == 1:
            merged_path = files_to_merge[0]
            needs_cleanup = False
        else:
            # Merge files first
            merged_path = os.path.join(temp_dir, "merged_temp.parquet")
            merge_parquet_files(files_to_merge, merged_path, batch_size=window_size)
            needs_cleanup = True

        if merge_only:
            # Just copy merged file to output
            if merged_path != output_path:
                if needs_cleanup:
                    shutil.move(merged_path, output_path)
                else:
                    shutil.copy2(merged_path, output_path)
            print(f"Merged file saved to: {output_path}")
        else:
            # Shuffle the merged file
            recursive_shuffle_parquet(
                merged_path,
                output_path,
                window_size=window_size,
                temp_dir=temp_dir,
                seed=seed,
                iterations=iterations,
                write_every_iteration=write_every_iteration,
            )

            # Clean up merged temp file if created
            if needs_cleanup and os.path.exists(merged_path):
                try:
                    os.remove(merged_path)
                except Exception:
                    pass

    finally:
        if cleanup_temp:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge and/or shuffle Parquet files using a batch-based approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shuffle a single file (original behavior)
  python randomize_parquet.py --input_path input.parquet --output_path output.parquet

  # Merge multiple files and shuffle
  python randomize_parquet.py --input_paths file1.parquet file2.parquet file3.parquet --output_path output.parquet

  # Merge all parquet files in a directory and shuffle
  python randomize_parquet.py --input_dir /path/to/parquet/files --output_path output.parquet

  # Only merge files without shuffling
  python randomize_parquet.py --input_dir /path/to/files --output_path merged.parquet --merge_only

  # Merge and shuffle with multiple iterations
  python randomize_parquet.py --input_dir /path/to/files --output_path output.parquet --iterations 3
        """,
    )

    # Input options (mutually exclusive in practice, but we handle the logic)
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument("--input_path", help="Path to a single input Parquet file")
    input_group.add_argument(
        "--input_paths",
        nargs="+",
        help="Paths to multiple input Parquet files to merge",
    )
    input_group.add_argument(
        "--input_dir", help="Directory containing Parquet files to merge"
    )

    # Output options
    parser.add_argument(
        "--output_path", required=True, help="Path for the output Parquet file"
    )

    # Processing options
    parser.add_argument(
        "--window-size",
        type=int,
        default=50000,
        help="Number of rows to process in each batch (default: 50000)",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory to store temporary files (default: system temp directory)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to repeat the shuffling process (default: 1)",
    )
    parser.add_argument(
        "--write_every_iteration",
        action="store_true",
        default=False,
        help="Save intermediate results after each shuffle iteration",
    )

    # Mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--merge_only",
        action="store_true",
        default=False,
        help="Only merge files without shuffling",
    )
    mode_group.add_argument(
        "--shuffle_only",
        action="store_true",
        default=False,
        help="Skip merging, only shuffle (requires --input_path)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.shuffle_only and not args.input_path:
        parser.error("--shuffle_only requires --input_path")

    if not args.input_path and not args.input_paths and not args.input_dir:
        parser.error(
            "Must provide at least one of: --input_path, --input_paths, or --input_dir"
        )

    merge_and_shuffle_parquet(
        input_paths=args.input_paths,
        input_dir=args.input_dir,
        output_path=args.output_path,
        window_size=args.window_size,
        temp_dir=args.temp_dir,
        seed=args.seed,
        iterations=args.iterations,
        write_every_iteration=args.write_every_iteration,
        merge_only=args.merge_only,
        shuffle_only=args.shuffle_only,
        input_path=args.input_path,
    )
