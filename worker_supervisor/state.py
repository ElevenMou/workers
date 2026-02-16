"""Shared logging state for worker supervisor modules."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s [%(process)d] %(levelname)-7s %(message)s",
)
logger = logging.getLogger("clipry.worker")
