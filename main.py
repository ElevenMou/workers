"""Backwards-compatible worker supervisor entrypoint."""

import sys

from worker_supervisor.supervisor import main


if __name__ == "__main__":
    sys.exit(main())
