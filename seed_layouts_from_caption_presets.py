#!/usr/bin/env python3
"""Create one layout per caption preset for a user."""

from __future__ import annotations

import argparse
from typing import Any

from services.caption_renderer import list_caption_presets
from utils.supabase_client import assert_response_ok, supabase

_VIDEO = {"widthPct": 100, "positionY": "middle"}
_TITLE = {
    "show": True,
    "fontSize": 48,
    "fontColor": "white",
    "fontFamily": "Montserrat",
    "align": "left",
    "strokeWidth": 0,
    "strokeColor": "black",
    "barEnabled": True,
    "barColor": "black@0.5",
    "paddingX": 16,
    "positionY": "above_video",
}


def _load_existing(user_id: str, name_prefix: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("layouts")
        .select("id,name")
        .eq("user_id", user_id)
        .ilike("name", f"{name_prefix}%")
        .execute()
    )
    assert_response_ok(resp, "Failed to load existing preset layouts")
    return list(resp.data or [])


def _delete_existing(rows: list[dict[str, Any]]) -> int:
    ids = [row.get("id") for row in rows if row.get("id")]
    if not ids:
        return 0
    resp = supabase.table("layouts").delete().in_("id", ids).execute()
    assert_response_ok(resp, "Failed to delete existing preset layouts")
    return len(ids)


def _build_layout_rows(user_id: str, name_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preset in list_caption_presets():
        label = str(preset.get("label") or preset.get("id") or "Preset").strip()
        captions = dict(preset.get("captions") or {})
        captions["show"] = True
        rows.append(
            {
                "user_id": user_id,
                "name": f"{name_prefix}{label}",
                "background_style": "solid_color",
                "background_color": "#101418",
                "background_blur_strength": 20,
                "video": _VIDEO,
                "title": _TITLE,
                "captions": captions,
                "output_quality": "high",
            }
        )
    return rows


def seed_layouts(user_id: str, name_prefix: str, dry_run: bool) -> tuple[int, int]:
    existing = _load_existing(user_id, name_prefix)
    deleted = len(existing)
    rows = _build_layout_rows(user_id, name_prefix)
    if dry_run:
        return deleted, len(rows)

    if existing:
        _delete_existing(existing)
    if rows:
        insert_resp = supabase.table("layouts").insert(rows).execute()
        assert_response_ok(insert_resp, "Failed to insert preset layouts")
    return deleted, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Create one layout for every caption preset for a user."
    )
    parser.add_argument("--user-id", required=True, help="Supabase user id")
    parser.add_argument(
        "--name-prefix",
        default="CC-Preset: ",
        help="Layout name prefix to replace/recreate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts only; do not modify DB",
    )
    args = parser.parse_args()

    deleted, created = seed_layouts(
        user_id=args.user_id,
        name_prefix=args.name_prefix,
        dry_run=args.dry_run,
    )
    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"{mode}: deleted={deleted} created={created}")


if __name__ == "__main__":
    main()
