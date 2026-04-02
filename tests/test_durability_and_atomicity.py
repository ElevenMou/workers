from __future__ import annotations

from pathlib import Path

from worker_supervisor import startup


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _combined_migration_sql() -> str:
    migration_dir = _repo_root() / "supabase" / "migrations"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(migration_dir.glob("*.sql"))
    )


def test_startup_queue_purge_is_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("PURGE_QUEUED_JOBS_ON_START", raising=False)
    caplog.set_level("INFO")

    startup.purge_startup_backlog(conn=object(), queue_names=["video-processing"])
    assert "Startup queue purge disabled" in caplog.text


def test_charge_credits_sql_uses_atomic_balance_update():
    sql = _combined_migration_sql()
    assert "UPDATE credit_balances" in sql
    assert "AND balance >= p_amount" in sql
    assert "RETURNING balance INTO v_balance_after" in sql


def test_team_wallet_reservation_sql_exists():
    sql = _combined_migration_sql()
    assert "CREATE TABLE IF NOT EXISTS public.team_wallet_reservations" in sql
    assert "CREATE OR REPLACE FUNCTION public.reserve_team_credits(" in sql
    assert "CREATE OR REPLACE FUNCTION public.capture_team_credit_reservation(" in sql
    assert "CREATE OR REPLACE FUNCTION public.release_team_credit_reservation(" in sql


def test_clip_tasks_finalize_before_charging_credits():
    generate_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "generate.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "custom.py"
    ).read_text(encoding="utf-8")

    assert generate_code.index("clip_update = {") < min(
        generate_code.index("capture_credit_reservation("),
        generate_code.index("capture_team_credit_reservation("),
    )
    assert custom_code.index("clip_update = {") < min(
        custom_code.index("capture_credit_reservation("),
        custom_code.index("capture_team_credit_reservation("),
    )


def test_clip_tasks_reserve_before_rendering():
    generate_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "generate.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "custom.py"
    ).read_text(encoding="utf-8")

    assert min(
        generate_code.index("reserve_credits("),
        generate_code.index("reserve_team_credits("),
    ) < generate_code.index("generator.generate(")
    assert min(
        custom_code.index("reserve_credits("),
        custom_code.index("reserve_team_credits("),
    ) < custom_code.index("generator.generate(")


def test_clip_tasks_emit_stage_progress_updates():
    generate_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "generate.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "custom.py"
    ).read_text(encoding="utf-8")

    generate_stages = [
        '"starting"',
        '"loading_clip"',
        '"loading_layout"',
        '"preparing_source_video"',
        '"preparing_captions"',
        '"rendering_clip"',
        '"uploading_clip"',
        '"charging_credits"',
        '"finalizing"',
    ]
    for stage in generate_stages:
        assert stage in generate_code

    custom_stages = [
        '"starting"',
        '"downloading_video"',
        '"loading_layout"',
        '"preparing_captions"',
        '"rendering_clip"',
        '"uploading_clip"',
        '"charging_credits"',
        '"finalizing"',
    ]
    for stage in custom_stages:
        assert stage in custom_code


def test_clip_uploads_set_media_content_types():
    generate_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "generate.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "custom.py"
    ).read_text(encoding="utf-8")

    for code in (generate_code, custom_code):
        assert '"content-type": "video/mp4"' in code


def test_clip_tasks_resolve_default_layout_before_rendering():
    generate_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "generate.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "custom.py"
    ).read_text(encoding="utf-8")

    for code in (generate_code, custom_code):
        assert "resolve_effective_layout_id(" in code
        assert code.index("resolve_effective_layout_id(") < code.index("load_layout_overrides(")
        assert 'clip_update["layout_id"] = layout_id' in code
