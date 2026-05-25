"""Publish parquet / JSON exports to Cloudflare R2 (S3-compatible).

KV is for small JSON the Worker reads on every request. R2 is for the
big stuff — parquet exports, agent traces, sharded signal dumps.

Env required:
  CLOUDFLARE_ACCOUNT_ID
  CF_R2_ACCESS_KEY_ID
  CF_R2_SECRET_ACCESS_KEY
  CF_R2_BUCKET            (e.g. rag-financial-insights)
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd


def _client():
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed — run: pip install boto3") from e

    account = os.environ.get("CF_R2_ACCOUNT_ID") or os.environ["CLOUDFLARE_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _bucket() -> str:
    name = os.environ.get("CF_R2_BUCKET")
    if not name:
        raise RuntimeError("CF_R2_BUCKET not set")
    return name


def upload_parquet(df: pd.DataFrame, key: str) -> str:
    """Write a DataFrame as parquet at `key`. Returns the s3:// URI."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _client().put_object(
        Bucket=_bucket(), Key=key, Body=buf.read(),
        ContentType="application/octet-stream",
    )
    return f"s3://{_bucket()}/{key}"


def upload_json(payload: dict | list, key: str) -> str:
    _client().put_object(
        Bucket=_bucket(), Key=key,
        Body=json.dumps(payload, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{_bucket()}/{key}"


def upload_file(local_path: Path, key: str) -> str:
    _client().upload_file(str(local_path), _bucket(), key)
    return f"s3://{_bucket()}/{key}"


def publish_cache_as_parquet(
    cache_dir: Path | None = None,
    key: str = "exports/signals.parquet",
) -> str:
    """Dump the local signal cache as one parquet file to R2."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "_signal_cache"
    rows = [json.loads(p.read_text()) for p in sorted(cache_dir.glob("*.json"))]
    if not rows:
        raise RuntimeError(f"no signals in {cache_dir}")
    uri = upload_parquet(pd.DataFrame(rows), key)
    print(f"Uploaded {len(rows)} signals → {uri}")
    return uri


if __name__ == "__main__":
    publish_cache_as_parquet()
