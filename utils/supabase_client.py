from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

# Monkey-patch gotrue/httpx proxy argument mismatch (httpx.Client expects "proxies")
try:
    import httpx
    from gotrue._sync import gotrue_base_api as _gotrue_base_api
    from gotrue import http_clients as _gotrue_http_clients

    class _PatchedSyncClient(httpx.Client):
        def __init__(self, *args, proxy=None, **kwargs):
            if proxy is not None and "proxies" not in kwargs:
                kwargs["proxies"] = proxy
            super().__init__(*args, **kwargs)

        def aclose(self) -> None:
            self.close()

    _gotrue_http_clients.SyncClient = _PatchedSyncClient  # type: ignore[attr-defined]

    # Override GoTrue base API to avoid passing unsupported "proxy" kw to httpx.Client
    def _patched_gotrue_init(self, *, url, headers, http_client, verify=True, proxy=None):
        self._url = url
        self._headers = headers
        self._http_client = http_client or _PatchedSyncClient(
            verify=bool(verify),
            proxies=proxy if proxy is not None else None,
            follow_redirects=True,
            http2=True,
        )

    _gotrue_base_api.SyncGoTrueBaseAPI.__init__ = _patched_gotrue_init  # type: ignore[assignment]
except Exception:
    # If patch fails, let the original error surface; this keeps startup resilient.
    pass

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def update_job_status(job_id: str, status: str, progress: int, error: str = None):
    """Update job status and progress"""
    data = {"status": status, "progress": progress}

    if status == "processing" and progress == 0:
        data["started_at"] = "now()"
    elif status in ["completed", "failed"]:
        data["completed_at"] = "now()"

    if error:
        data["error_message"] = error

    supabase.table("jobs").update(data).eq("id", job_id).execute()


def update_video_status(video_id: str, status: str, **kwargs):
    """Update video record"""
    data = {"status": status}
    data.update(kwargs)
    supabase.table("videos").update(data).eq("id", video_id).execute()
