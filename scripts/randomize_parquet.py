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
            print(f"Starting shuffle iteration {iteration+1}/{iterations}")

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
            for i, batch in enumerate(tqdm(
                current_file.iter_batches(batch_size=window_size),
                desc="Processing batches",
                total=(total_rows + window_size - 1) // window_size
            )):
                # Convert to pandas, shuffle, and convert back
                df = batch.to_pandas()
                df = df.sample(frac=1.0)

                # Create a temporary file for this batch
                batch_file = os.path.join(temp_dir, f"batch_{i}_{random.randint(1000000, 9999999)}.parquet")

                # Use compression to save disk space
                table = pa.Table.from_pandas(df, schema=current_schema)
                pq.write_table(table, batch_file, compression='snappy')

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
            with pq.ParquetWriter(iter_output, current_schema, compression='snappy') as writer:
                for batch_file in tqdm(batch_files, desc="Combining batches"):
                    try:
                        # Read this batch, write to output, and immediately delete
                        table = pq.read_table(batch_file)
                        table = table.cast(current_schema)  # Ensure schema compatibility
                        writer.write_table(table)

                        # Clean up memory and disk
                        del table
                        os.remove(batch_file)
                    except Exception as e:
                        print(f"Warning: Error processing batch file {batch_file}: {e}")

            # Update for next iteration
            if iteration < iterations - 1:
                current_input = iter_output
            else:
                # Final iteration - copy to output_path
                shutil.copy2(iter_output, output_path)

                # Clean up the final iteration file
                try:
                    os.remove(iter_output)
                except Exception as e:
                    print(f"Warning: Could not remove final iteration file: {e}")

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
