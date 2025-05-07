import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import tempfile
import shutil
from tqdm import tqdm
import random

def recursive_shuffle_parquet(input_path, output_path, window_size=50000, temp_dir=None, seed=None):
    """
    Thoroughly shuffle a Parquet file using a recursive, out-of-core approach.

    Parameters:
    -----------
    input_path : str
        Path to the input Parquet file
    output_path : str
        Path for the output shuffled Parquet file
    window_size : int
        Number of rows to process in each window/buffer
    temp_dir : str, optional
        Directory to store temporary files. If None, a system temp directory is used.
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create a temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)

    first_level_dir = os.path.join(temp_dir, "level1")
    second_level_dir = os.path.join(temp_dir, "level2")
    os.makedirs(first_level_dir, exist_ok=True)
    os.makedirs(second_level_dir, exist_ok=True)

    try:
        print("Reading Parquet file metadata...")
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        schema = parquet_file.schema_arrow if hasattr(parquet_file, 'schema_arrow') else parquet_file.schema

        print(f"Total rows in the file: {total_rows}")
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: First level of shuffling - divide into M pieces of size N
        print("Step 1: First level shuffling - creating initial chunks...")
        m_chunks = []
        chunk_count = 0

        # Process the file in windows and shuffle each window
        for batch in tqdm(parquet_file.iter_batches(batch_size=window_size),
                         desc="Creating shuffled chunks",
                         total=(total_rows + window_size - 1) // window_size):
            df = batch.to_pandas()

            # Shuffle the current window
            df = df.sample(frac=1.0)

            # Write to a temporary file
            chunk_path = os.path.join(first_level_dir, f"chunk_{chunk_count}.parquet")
            df.to_parquet(chunk_path, index=False)
            m_chunks.append(chunk_path)
            chunk_count += 1

        # Step 2: Second level of shuffling - combine M readers randomly
        print(f"Step 2: Second level shuffling - combining {len(m_chunks)} chunks...")

        # For each chunk, divide it into smaller pieces for random access
        chunk_pieces = []
        for i, chunk_path in enumerate(m_chunks):
            # Load the entire chunk
            df = pd.read_parquet(chunk_path)
            # Divide into smaller pieces for random access
            piece_size = min(window_size // 10, len(df))  # Use smaller pieces
            if piece_size <= 0:  # Handle very small dataframes
                piece_size = len(df)

            # Divide the dataframe into pieces and store each piece
            for j in range(0, len(df), piece_size):
                end = min(j + piece_size, len(df))
                piece = df.iloc[j:end]
                piece_path = os.path.join(first_level_dir, f"chunk_{i}_piece_{j//piece_size}.parquet")
                piece.to_parquet(piece_path, index=False)
                chunk_pieces.append(piece_path)

        # Shuffle the order of pieces to randomize access
        random.shuffle(chunk_pieces)

        # Calculate total number of rows across all pieces
        total_shuffled_rows = sum(pd.read_parquet(piece).shape[0] for piece in chunk_pieces)

        # Create second level shuffle files
        second_level_chunks = []
        second_chunk_count = 0

        # Continue until we've processed all rows
        rows_processed = 0
        remaining_pieces = chunk_pieces.copy()
        pbar = tqdm(total=total_shuffled_rows, desc="Creating second-level shuffle")

        while remaining_pieces and rows_processed < total_shuffled_rows:
            # Create a new output chunk
            output_chunk_path = os.path.join(second_level_dir, f"shuffled_{second_chunk_count}.parquet")
            second_level_chunks.append(output_chunk_path)

            # Determine how many rows to take for this chunk
            rows_for_chunk = min(window_size, total_shuffled_rows - rows_processed)

            # Collect rows from random pieces until we have enough
            collected_dfs = []
            collected_rows_count = 0

            while collected_rows_count < rows_for_chunk and remaining_pieces:
                # Choose a random piece
                piece_idx = random.randrange(len(remaining_pieces))
                piece_path = remaining_pieces.pop(piece_idx)

                # Read the piece
                df_piece = pd.read_parquet(piece_path)

                # Add to our collection
                collected_dfs.append(df_piece)
                collected_rows_count += len(df_piece)

            if collected_dfs:
                # Combine collected rows and shuffle again
                combined_df = pd.concat(collected_dfs, ignore_index=True)
                combined_df = combined_df.sample(frac=1.0)  # Shuffle once more

                # If we have more rows than needed, take only what we need
                if len(combined_df) > rows_for_chunk:
                    combined_df = combined_df.iloc[:rows_for_chunk]

                # Write to output chunk
                combined_df.to_parquet(output_chunk_path, index=False)

                rows_processed += len(combined_df)
                pbar.update(len(combined_df))
                second_chunk_count += 1

        pbar.close()

        # Step 3: Combine all second-level chunks into the final output file
        print("Step 3: Combining all second-level shuffled chunks into final output...")

        # Read and combine all chunks
        all_dfs = []
        for chunk_path in tqdm(second_level_chunks, desc="Reading shuffled chunks"):
            df = pd.read_parquet(chunk_path)
            all_dfs.append(df)

        final_df = pd.concat(all_dfs, ignore_index=True)

        # Write the final output
        print(f"Writing final file with {len(final_df)} rows...")
        final_df.to_parquet(output_path, index=False)

        print(f"Shuffled parquet file saved to: {output_path}")

    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recursively shuffle a Parquet file out-of-core")
    parser.add_argument("--input_path", help="Path to the input Parquet file")
    parser.add_argument("--output_path", help="Path for the output shuffled Parquet file")
    parser.add_argument("--window-size", type=int, default=50000,
                        help="Number of rows to process in each window (default: 50000)")
    parser.add_argument("--temp-dir", default=None,
                        help="Directory to store temporary files (default: system temp directory)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    recursive_shuffle_parquet(args.input_path, args.output_path,
                             window_size=args.window_size,
                             temp_dir=args.temp_dir,
                             seed=args.seed)
