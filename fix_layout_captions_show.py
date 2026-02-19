#!/usr/bin/env python3
"""Enable captions.show for saved layouts that already use a caption preset."""

from __future__ import annotations

import argparse
from typing import Any

from utils.supabase_client import assert_response_ok, supabase


def _should_enable(layout: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    captions = layout.get("captions")
    if not isinstance(captions, dict):
        return False, {}

    preset_name = (
        captions.get("presetName")
        or captions.get("preset")
        or captions.get("style")
    )
    if not preset_name:
        return False, captions

    if captions.get("show") is True:
        return False, captions

    updated = dict(captions)
    updated["show"] = True
    return True, updated


def enable_layout_captions(
    *,
    user_id: str,
    name_prefix: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    query = (
        supabase.table("layouts")
        .select("id,name,captions")
        .eq("user_id", user_id)
        .order("created_at")
    )
    if name_prefix:
        query = query.ilike("name", f"{name_prefix}%")

    resp = query.execute()
    assert_response_ok(resp, "Failed to fetch layouts")
    layouts = list(resp.data or [])

    inspected = len(layouts)
    updated_count = 0

    for layout in layouts:
        layout_id = layout.get("id")
        if not layout_id:
            continue

        should_enable, updated_captions = _should_enable(layout)
        if not should_enable:
            continue

        updated_count += 1
        if dry_run:
            continue

        update_resp = (
            supabase.table("layouts")
            .update({"captions": updated_captions})
            .eq("id", layout_id)
            .execute()
        )
        assert_response_ok(update_resp, f"Failed to update layout {layout_id}")

    return inspected, updated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set captions.show=true for existing layouts that already have a "
            "caption preset configured."
        )
    )
    parser.add_argument("--user-id", required=True, help="Supabase user id")
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Optional layout name prefix filter (for preset batches).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts; do not persist updates.",
    )
    args = parser.parse_args()

    inspected, updated = enable_layout_captions(
        user_id=args.user_id,
        name_prefix=args.name_prefix or None,
        dry_run=args.dry_run,
    )
    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"{mode}: inspected={inspected} updated={updated}")


if __name__ == "__main__":
    main()

