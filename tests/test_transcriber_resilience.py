from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

if "whisper" not in sys.modules:
    whisper_stub = ModuleType("whisper")
    whisper_stub._MODELS = {}
    whisper_stub.load_model = lambda *_args, **_kwargs: SimpleNamespace(
        transcribe=lambda *_a, **_k: {"text": "", "segments": [], "language": "en"}
    )
    sys.modules["whisper"] = whisper_stub

from services import transcriber as transcriber_module


class _NoOpLock:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_get_model_retries_after_checksum_mismatch_and_clears_cache(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "whisper-cache"
    corrupt_model_path = cache_dir / "tiny.pt"
    corrupt_model_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_model_path.write_bytes(b"corrupt")

    load_calls = {"count": 0}
    loaded_model = object()

    def _fake_load_model(model_name: str, *, download_root: str | None = None):
        assert model_name == "tiny"
        assert download_root == str(cache_dir)
        load_calls["count"] += 1
        if load_calls["count"] == 1:
            assert corrupt_model_path.exists()
            raise RuntimeError(
                "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
            )
        return loaded_model

    monkeypatch.setenv("WHISPER_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(transcriber_module, "_models", {})
    monkeypatch.setattr(transcriber_module, "_InterProcessFileLock", _NoOpLock)
    monkeypatch.setattr(
        transcriber_module,
        "whisper",
        SimpleNamespace(
            _MODELS={"tiny": "https://example.com/expected-sha256/tiny.pt"},
            load_model=_fake_load_model,
        ),
    )

    model = transcriber_module._get_model("tiny")

    assert model is loaded_model
    assert load_calls["count"] == 2
    assert transcriber_module._models["tiny"] is loaded_model
    assert corrupt_model_path.exists() is False
