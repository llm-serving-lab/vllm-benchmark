"""Upload benchmark results (runs/*.csv, runs/*.jsonl) to an S3-compatible bucket.

Replaces the MinIO `mc cp` step in run_and_upload.sh. Same behavior as
before -- every CSV/JSONL in runs/ goes to the bucket under its own file
name -- only the client changed. `mc` was downloaded from dl.min.io, which
now returns HTTP 410 (MinIO's community distribution was shut down);
boto3 comes from PyPI instead.

Configuration (all from the environment, never hard-coded):
  S3_ENDPOINT            e.g. http://garage:3900
  S3_BUCKET              e.g. bench-results
  AWS_ACCESS_KEY_ID      read by boto3 automatically
  AWS_SECRET_ACCESS_KEY  read by boto3 automatically
  AWS_DEFAULT_REGION     Garage's default region name is "garage"
"""

import glob
import os
import sys

import boto3
from botocore.config import Config


def main() -> int:
    endpoint = os.environ["S3_ENDPOINT"]
    bucket = os.environ["S3_BUCKET"]

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            # Path-style URLs (http://host/bucket/key). Virtual-host style
            # (http://bucket.host/key) needs DNS per bucket, which an
            # in-cluster Service name doesn't have.
            s3={"addressing_style": "path"},
            # Newer boto3 versions add extra checksum headers by default;
            # many S3-compatible stores don't accept them. Only send them
            # when an operation actually requires it.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )

    files = sorted(glob.glob("runs/*.csv") + glob.glob("runs/*.jsonl"))
    if not files:
        print("No result files found in runs/", file=sys.stderr)
        return 1

    for path in files:
        key = os.path.basename(path)
        s3.upload_file(path, bucket, key)
        print(f"uploaded {path} -> s3://{bucket}/{key}")
    print(f"Uploaded {len(files)} files to {endpoint}/{bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
