from pathlib import Path

import boto3
import typer
from botocore import UNSIGNED
from botocore.config import Config


def download_s3_bucket(
    bucket_name: str,
    endpoint: str = "https://storage.yandexcloud.net",
    prefix: str = "",
    local_dir: Path | None = None
) -> None:
    """Download files from S3 bucket to local directory.

    Args:
        bucket_name: Name of the S3 bucket to download from.
        endpoint: S3 endpoint URL.
        prefix: Prefix to filter objects in the bucket.
        local_dir: Local directory to save files. If None, uses ../data
            relative to current directory.
    """
    # Default local directory
    if local_dir is None:
        local_dir = Path.cwd().parent / "data"

    # Create S3 client without authorization
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(signature_version=UNSIGNED)
    )

    # Ensure local directory exists
    local_dir.mkdir(parents=True, exist_ok=True)

    # Get paginator for list_objects_v2
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Calculate local path
            rel_path = key
            local_path = local_dir / rel_path

        # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            print(f"Downloading {key} → {local_path}")
            s3.download_file(bucket_name, key, str(local_path))


def main(
    set_name: str = typer.Argument(
        ...,
        help="Name of the set to download.",
        metavar="SET"
    )
) -> None:
    """Download files from S3 bucket to local directory.

    Download either the 'train-val' or 'test' dataset from the S3 bucket.
    Files will be saved to the data directory relative to the script location.
    """
    # Validate the set name
    if set_name not in ["train-val", "test"]:
        error_msg = (
            f"Error: Invalid set name '{set_name}'. "
            "Must be 'train-val' or 'test'."
        )
        typer.echo(error_msg, err=True)
        raise typer.Exit(1)

    bucket_name = "itlp-campus-seq-data"
    local_dir = Path(__file__).parent.parent / "data"

    download_s3_bucket(
        bucket_name=bucket_name, prefix=set_name, local_dir=local_dir
    )


if __name__ == "__main__":
    typer.run(main)
