"""Backwards-compatible API entrypoint."""

from api_app import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
