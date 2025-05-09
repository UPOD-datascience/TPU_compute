import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import tempfile
import shutil
from tqdm import tqdm
import random

def recursive_shuffle_parquet(input_path, output_path, window_size=50000, temp_dir=None, seed=None, iterations=1):
    """
    Shuffles a Parquet file using a simpler, batch-based approach optimized for low memory usage.

    Process:
    1. Read batches from input file and shuffle each batch internally
    2. Write each shuffled batch to a temporary file
    3. Shuffle the order of batch files
    4. Recombine the batches in shuffled order to the output file
    5. Repeat the process for specified number of iterations

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

        # Extract the original schema to maintain consistency
        original_schema = parquet_file.schema_arrow
        print(f"Original schema: {original_schema}")

        print(f"Total rows in the file: {total_rows}")
        print(f"Using temporary directory: {temp_dir}")

        # Use the input path for first iteration, then use intermediate outputs for subsequent iterations
        current_input = input_path

        for iteration in range(iterations):
            print(f"Starting shuffle iteration {iteration+1}/{iterations}")

            # Step 1: Read batches, shuffle each batch, and write to temporary files
            batch_files = []
            batch_count = 0

            # Create sub-directory for this iteration's batch files
            iter_temp_dir = os.path.join(temp_dir, f"iteration_{iteration}")
            os.makedirs(iter_temp_dir, exist_ok=True)

            print("Step 1: Reading and shuffling batches...")
            current_parquet_file = pq.ParquetFile(current_input)

            # For the first iteration, use the original schema
            # For subsequent iterations, get the schema from the current file
            if iteration == 0:
                current_schema = original_schema
            else:
                current_schema = current_parquet_file.schema_arrow

            # Keep track of total rows processed to ensure accurate progress
            actual_total_rows = 0

            # Process each batch
            for i, batch in enumerate(tqdm(
                current_parquet_file.iter_batches(batch_size=window_size),
                desc="Processing batches",
                total=(total_rows + window_size - 1) // window_size
            )):
                # Convert to pandas for efficient shuffling
                df = batch.to_pandas()
                actual_total_rows += len(df)

                # Shuffle the batch internally
                df = df.sample(frac=1.0)

                # Write to a temporary file
                batch_path = os.path.join(iter_temp_dir, f"batch_{batch_count}.parquet")

                # Convert back to PyArrow table with explicit schema to maintain consistency
                table = pa.Table.from_pandas(df, schema=current_schema)

                # Write with schema preservation
                pq.write_table(table, batch_path, compression='snappy')

                # Release memory by deleting references
                del df
                del table

                batch_files.append(batch_path)
                batch_count += 1

            # Update total rows for future iterations
            total_rows = actual_total_rows

            # Step 2: Shuffle the order of batch files
            print(f"Step 2: Shuffling order of {len(batch_files)} batch files...")
            random.shuffle(batch_files)

            # Step 3: Recombine batches in shuffled order
            print("Step 3: Recombining batches in shuffled order...")
            temp_output_path = os.path.join(temp_dir, f"temp_output_{iteration}.parquet")

            # Start with an empty table of the correct schema
            if batch_files:
                # Use the schema from the current iteration to maintain consistency
                with pq.ParquetWriter(temp_output_path, current_schema) as writer:
                    # Read each batch in shuffled order and write to the output file
                    for batch_path in tqdm(batch_files, desc="Recombining batches"):
                        # Read the entire file as a table to ensure schema compatibility
                        table = pq.read_table(batch_path)

                        # Cast the table to the required schema to ensure consistency
                        table = table.cast(current_schema)

                        # Write the table
                        writer.write_table(table)

                        # Release memory
                        del table
            else:
                # Handle the case where there are no batch files (unlikely but possible)
                print("Warning: No batch files were created. Creating an empty output file.")

                # Create an empty table with the same schema as the input
                empty_table = pa.Table.from_arrays(
                    [pa.array([], type=field.type) for field in current_schema],
                    schema=current_schema
                )
                pq.write_table(empty_table, temp_output_path)

            # Update the current input for the next iteration
            current_input = temp_output_path

            # Clean up this iteration's batch files
            for batch_path in batch_files:
                try:
                    os.remove(batch_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {batch_path}: {e}")

            # Clean up the iteration directory
            try:
                os.rmdir(iter_temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory {iter_temp_dir}: {e}")

        # After all iterations, move the final shuffled file to the output path
        shutil.copy2(current_input, output_path)

        print(f"Shuffled parquet file saved to: {output_path}")

    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")
        print("Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shuffle a Parquet file using a batch-based approach")
    parser.add_argument("--input_path", required=True, help="Path to the input Parquet file")
    parser.add_argument("--output_path", required=True, help="Path for the output shuffled Parquet file")
    parser.add_argument("--window-size", type=int, default=50000,
                        help="Number of rows to process in each batch (default: 50000)")
    parser.add_argument("--temp-dir", default=None,
                        help="Directory to store temporary files (default: system temp directory)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of times to repeat the shuffling process (default: 1)")

    args = parser.parse_args()

    recursive_shuffle_parquet(
        args.input_path,
        args.output_path,
        window_size=args.window_size,
        temp_dir=args.temp_dir,
        seed=args.seed,
        iterations=args.iterations
    )
