#!/usr/bin/env python3
"""
Script to upload parquet files to Google Cloud Storage (GCS).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

try:
    from google.cloud import storage
except ImportError:
    print("Error: google-cloud-storage is required. Install with: pip install google-cloud-storage")
    sys.exit(1)


def find_parquet_files(directory: str) -> List[Path]:
    """
    Find all parquet files in the specified directory.

    Args:
        directory: Path to the directory containing parquet files

    Returns:
        List of Path objects for parquet files
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    parquet_files = list(directory_path.glob("*.parquet"))
    if not parquet_files:
        print(f"Warning: No parquet files found in {directory}")

    return parquet_files


def upload_file_to_gcs(client: storage.Client, bucket_name: str, local_file_path: Path,
                      gcs_destination_path: str) -> bool:
    """
    Upload a single file to GCS.

    Args:
        client: GCS client instance
        bucket_name: Name of the GCS bucket
        local_file_path: Path to the local file
        gcs_destination_path: Destination path in GCS

    Returns:
        True if upload successful, False otherwise
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_destination_path)

        print(f"Uploading {local_file_path.name} to gs://{bucket_name}/{gcs_destination_path}")
        blob.upload_from_filename(str(local_file_path))
        print(f"✓ Successfully uploaded {local_file_path.name}")
        return True

    except Exception as e:
        print(f"✗ Failed to upload {local_file_path.name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload parquet files to Google Cloud Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script_to_upload_to_gcs.py /path/to/parquet/files my-bucket data/parquet/
  python script_to_upload_to_gcs.py ./data my-bucket processed/ --recursive
        """
    )

    parser.add_argument(
        "--location_of_parquet_files",
        help="Local directory containing parquet files to upload"
    )

    parser.add_argument(
        "--gcs_bucket_name",
        help="Name of the GCS bucket (without gs:// prefix)"
    )

    parser.add_argument(
        "--gcs_location",
        help="Destination path/prefix in GCS bucket (e.g., 'data/parquet/')"
    )

    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search for parquet files recursively in subdirectories"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )

    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve directory structure in GCS destination"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.location_of_parquet_files):
        print(f"Error: Directory does not exist: {args.location_of_parquet_files}")
        sys.exit(1)

    # Ensure GCS location ends with '/' if it's meant to be a directory
    gcs_location = args.gcs_location
    if gcs_location and not gcs_location.endswith('/'):
        gcs_location += '/'

    try:
        # Initialize GCS client
        client = storage.Client()

        # Test bucket access
        try:
            bucket = client.bucket(args.gcs_bucket_name)
            bucket.reload()  # This will raise an exception if bucket doesn't exist or no access
        except Exception as e:
            print(f"Error: Cannot access bucket '{args.gcs_bucket_name}': {str(e)}")
            print("Make sure the bucket exists and you have proper authentication.")
            sys.exit(1)

        # Find parquet files
        directory_path = Path(args.location_of_parquet_files)

        if args.recursive:
            parquet_files = list(directory_path.rglob("*.parquet"))
        else:
            parquet_files = list(directory_path.glob("*.parquet"))

        if not parquet_files:
            print(f"No parquet files found in {args.location_of_parquet_files}")
            sys.exit(0)

        print(f"Found {len(parquet_files)} parquet file(s)")

        if args.dry_run:
            print("\nDry run - would upload the following files:")
            for file_path in parquet_files:
                if args.preserve_structure:
                    relative_path = file_path.relative_to(directory_path)
                    gcs_path = gcs_location + str(relative_path)
                else:
                    gcs_path = gcs_location + file_path.name
                print(f"  {file_path} -> gs://{args.gcs_bucket_name}/{gcs_path}")
            return

        # Upload files
        successful_uploads = 0
        failed_uploads = 0

        for file_path in parquet_files:
            if args.preserve_structure:
                # Preserve directory structure
                relative_path = file_path.relative_to(directory_path)
                gcs_destination = gcs_location + str(relative_path)
            else:
                # Just use filename
                gcs_destination = gcs_location + file_path.name

            if upload_file_to_gcs(client, args.gcs_bucket_name, file_path, gcs_destination):
                successful_uploads += 1
            else:
                failed_uploads += 1

        # Summary
        print(f"\nUpload complete:")
        print(f"  ✓ Successful: {successful_uploads}")
        print(f"  ✗ Failed: {failed_uploads}")

        if failed_uploads > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nUpload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
