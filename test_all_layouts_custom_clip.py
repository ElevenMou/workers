#!/usr/bin/env python3
"""Queue one custom clip job per saved layout for a user."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx

from utils.supabase_client import assert_response_ok, supabase

DEFAULT_API_URL = "http://localhost:8001"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=6NwDKDzMRNY"
DEFAULT_USER_ID = "f0cae5c7-7386-4174-ac43-af9a4dc233a0"
DEFAULT_START_TIME = 0.0
DEFAULT_END_TIME = 45.0


def _build_title(layout: dict[str, Any]) -> str:
    """Use layout name as clip title, with safe fallbacks."""
    name = str(layout.get("name") or "").strip()
    if name:
        return name[:200]

    title_cfg = layout.get("title")
    if isinstance(title_cfg, dict):
        for key in ("text", "name", "label"):
            value = str(title_cfg.get(key) or "").strip()
            if value:
                return value[:200]

    return f"Layout {layout.get('id', 'unknown')}"[:200]


def _validate_clip_range(start_time: float, end_time: float):
    if end_time <= start_time:
        raise ValueError("endTime must be greater than startTime")

    duration = end_time - start_time
    if duration < 10:
        raise ValueError(
            f"Invalid clip duration {duration:.2f}s. Must be at least 10 seconds."
        )


def _fetch_layouts(user_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("layouts")
        .select("id,name,title,created_at")
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    assert_response_ok(resp, "Failed to fetch layouts")
    return list(resp.data or [])


def _queue_layout_clip(
    *,
    client: httpx.Client,
    api_url: str,
    access_token: str,
    video_url: str,
    layout: dict[str, Any],
    start_time: float,
    end_time: float,
) -> tuple[bool, str]:
    payload = {
        "url": video_url,
        "startTime": start_time,
        "endTime": end_time,
        "title": _build_title(layout),
        "layoutId": layout["id"],
    }
    response = client.post(
        f"{api_url.rstrip('/')}/clips/custom",
        json=payload,
        headers={"Authorization": f"Bearer {access_token}"},
    )

    if response.is_success:
        data = response.json()
        job_id = data.get("jobId", "?")
        clip_id = data.get("clipId", "?")
        return True, f"queued jobId={job_id} clipId={clip_id}"

    try:
        error_payload = response.json()
        error_text = json.dumps(error_payload)
    except Exception:
        error_text = response.text or "<empty response body>"

    return False, f"HTTP {response.status_code} {error_text}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Queue one custom clip generation request per saved layout for a user."
        )
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--access-token", required=True)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--start-time", type=float, default=DEFAULT_START_TIME)
    parser.add_argument("--end-time", type=float, default=DEFAULT_END_TIME)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of layouts to test (0 = all).",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests to avoid queue bursts.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout per request.",
    )
    args = parser.parse_args()

    _validate_clip_range(args.start_time, args.end_time)
    layouts = _fetch_layouts(args.user_id)

    if not layouts:
        print(f"No layouts found for user {args.user_id}")
        return

    if args.limit > 0:
        layouts = layouts[: args.limit]

    print(
        f"Testing {len(layouts)} layouts for user={args.user_id} "
        f"video={args.video_url} range={args.start_time:.2f}-{args.end_time:.2f}"
    )

    success_count = 0
    failures: list[tuple[str, str, str]] = []

    with httpx.Client(timeout=args.timeout_seconds) as client:
        for index, layout in enumerate(layouts, start=1):
            layout_id = str(layout.get("id"))
            layout_name = str(layout.get("name") or "").strip() or "<unnamed layout>"
            title = _build_title(layout)

            try:
                ok, detail = _queue_layout_clip(
                    client=client,
                    api_url=args.api_url,
                    access_token=args.access_token,
                    video_url=args.video_url,
                    layout=layout,
                    start_time=args.start_time,
                    end_time=args.end_time,
                )
            except Exception as exc:
                ok = False
                detail = f"Request exception: {exc}"

            prefix = "OK" if ok else "FAIL"
            print(f"[{index}/{len(layouts)}] {prefix} {layout_name} ({layout_id}) -> {detail}")

            if ok:
                success_count += 1
            else:
                failures.append((layout_id, layout_name, detail))

            if args.delay_seconds > 0 and index < len(layouts):
                time.sleep(args.delay_seconds)

    print(
        f"Done. success={success_count} failed={len(failures)} total={len(layouts)}"
    )

    if failures:
        print("\nFailed layouts:")
        for layout_id, layout_name, detail in failures:
            print(f"- {layout_name} ({layout_id}): {detail}")


if __name__ == "__main__":
    main()
