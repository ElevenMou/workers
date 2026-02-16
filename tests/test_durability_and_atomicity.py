from __future__ import annotations

from pathlib import Path

from worker_supervisor import startup


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_startup_queue_purge_is_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("PURGE_QUEUED_JOBS_ON_START", raising=False)
    caplog.set_level("INFO")

    startup.purge_startup_backlog(conn=object(), queue_names=["video-processing"])
    assert "Startup queue purge disabled" in caplog.text


def test_charge_credits_sql_uses_atomic_balance_update():
    sql = (_repo_root() / "SQL" / "transactions.sql").read_text(encoding="utf-8")
    assert "UPDATE credit_balances" in sql
    assert "AND balance >= p_amount" in sql
    assert "RETURNING balance INTO v_balance_after" in sql


def test_clip_tasks_charge_before_marking_completed():
    generate_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "generate.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        _repo_root() / "workers" / "tasks" / "clips" / "custom.py"
    ).read_text(encoding="utf-8")

    assert generate_code.index("charge_clip_generation_credits(") < generate_code.index(
        "clip_update = {"
    )
    assert custom_code.index("charge_clip_generation_credits(") < custom_code.index(
        "clip_update = {"
    )
