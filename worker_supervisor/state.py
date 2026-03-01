"""Shared logging state for worker supervisor modules."""

import logging

from utils.logging_config import setup_logging

setup_logging(component="worker")
logger = logging.getLogger("clipry.worker")
