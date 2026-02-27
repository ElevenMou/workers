"""Simple API enqueue load harness for staging validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from typing import Any

import requests


def _post_json(
    *,
    base_url: str,
    token: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int, str]:
    response = requests.post(
        f"{base_url.rstrip('/')}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=timeout_seconds,
    )
    body = response.text[:500]
    return response.status_code, body


def main() -> int:
    parser = argparse.ArgumentParser(description="Clipry workers load harness")
    parser.add_argument("--base-url", required=True, help="API base URL, for example http://localhost:8001")
    parser.add_argument("--token", required=True, help="Supabase JWT for authenticated routes")
    parser.add_argument("--path", default="/videos/analyze", help="Route to exercise")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--payload-json",
        required=True,
        help="JSON string payload sent for each request",
    )
    args = parser.parse_args()

    payload = json.loads(args.payload_json)
    start = time.time()
    statuses: dict[int, int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                _post_json,
                base_url=args.base_url,
                token=args.token,
                path=args.path,
                payload=payload,
                timeout_seconds=args.timeout,
            )
            for _ in range(max(1, args.requests))
        ]

        for fut in concurrent.futures.as_completed(futures):
            status_code, _body = fut.result()
            statuses[status_code] = statuses.get(status_code, 0) + 1

    elapsed = max(0.001, time.time() - start)
    rps = args.requests / elapsed
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "elapsed_seconds": round(elapsed, 3),
                "requests_per_second": round(rps, 3),
                "status_counts": statuses,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
