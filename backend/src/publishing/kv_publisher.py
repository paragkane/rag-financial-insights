"""Publish signal snapshots to Cloudflare Workers KV.

Reads CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN from env, plus the KV
namespace id (default env: CF_KV_NAMESPACE_ID; matches the binding
configured in workers/wrangler.toml). Writes one key per ticker plus an
index key — the Workers route at /api/signals streams them to the SolidJS
dashboard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

CF_API = "https://api.cloudflare.com/client/v4"


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing env var: {name}")
    return val


def _put_key(account_id: str, namespace_id: str, key: str, value: Any, token: str) -> None:
    url = f"{CF_API}/accounts/{account_id}/storage/kv/namespaces/{namespace_id}/values/{key}"
    body = value if isinstance(value, str) else json.dumps(value, default=str)
    resp = requests.put(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=body,
        timeout=30,
    )
    resp.raise_for_status()


def publish_signals(signals: list[dict], namespace_id: str | None = None) -> int:
    """Write `signals` to KV, one key per ticker, plus a sorted index key.

    Returns the total number of keys written.
    """
    account_id = _env("CLOUDFLARE_ACCOUNT_ID")
    token = _env("CLOUDFLARE_API_TOKEN")
    namespace_id = namespace_id or _env("CF_KV_NAMESPACE_ID")

    by_ticker: dict[str, list[dict]] = {}
    for s in signals:
        by_ticker.setdefault(s["ticker"].upper(), []).append(s)

    written = 0
    for ticker, rows in by_ticker.items():
        rows.sort(key=lambda r: r.get("date", ""))
        _put_key(account_id, namespace_id, f"signals:{ticker}", rows, token)
        written += 1

    _put_key(account_id, namespace_id, "signals:index", sorted(by_ticker.keys()), token)
    return written + 1


def publish_from_cache(cache_dir: Path | None = None, namespace_id: str | None = None) -> int:
    """Walk the existing _signal_cache and publish everything found."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "_signal_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"No signal cache at {cache_dir}")

    signals = [json.loads(p.read_text()) for p in sorted(cache_dir.glob("*.json"))]
    if not signals:
        print("Nothing to publish.")
        return 0

    n = publish_signals(signals, namespace_id=namespace_id)
    print(f"Published {len(signals)} signals across {n} keys.")
    return n


if __name__ == "__main__":
    publish_from_cache()
