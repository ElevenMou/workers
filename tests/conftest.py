import os
import sys
import types

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault(
    "SUPABASE_SERVICE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJyb2xlIjoic2VydmljZV9yb2xlIiwiaXNzIjoic3VwYWJhc2UiLCJleHAiOjQxMDI0NDQ4MDB9."
    "test-signature",
)
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DISABLE_RATE_LIMITS", "true")
os.environ.setdefault("RATE_LIMIT_FAIL_OPEN", "true")
os.environ.setdefault("MEDIA_STORAGE_PROVIDER", "minio")
os.environ.setdefault("WORKER_PUBLIC_BASE_URL", "http://testserver")
os.environ.setdefault("WORKER_MEDIA_SIGNING_SECRET", "test-media-secret")
os.environ.setdefault("WORKER_INTERNAL_API_TOKEN", "test-internal-token")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_PUBLIC_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("MINIO_SKIP_STARTUP_READINESS", "true")

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_stub = types.ModuleType("openai")
    openai_stub.NotFoundError = type("NotFoundError", (Exception,), {})
    openai_stub.LengthFinishReasonError = type("LengthFinishReasonError", (Exception,), {})

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    openai_stub.OpenAI = _OpenAI
    sys.modules.setdefault("openai", openai_stub)

from api_app.app import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
