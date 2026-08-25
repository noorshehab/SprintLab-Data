"""Central logging setup for SprintLab services.

Usage:
    from services.log_setup import get_logger
    log = get_logger(__name__)
    log.info('...')

All loggers write to a rotating txt file (logs/sprintlab_log.txt, 5MB x 3)
and optionally to stderr. Level is controlled by the SPRINTLAB_LOG_LEVEL
env var (DEBUG/INFO/WARNING/ERROR; default INFO).
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'sprintlab_log.txt')
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
_configured = False


def _resolve_level() -> int:
    return getattr(logging, os.environ.get('SPRINTLAB_LOG_LEVEL', 'INFO').upper(), logging.INFO)


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger('sprintlab')
    root.setLevel(_resolve_level())
    root.propagate = False

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES,
                                       backupCount=BACKUP_COUNT, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file_handler)

    if os.environ.get('SPRINTLAB_LOG_CONSOLE', '') == '1':
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the shared 'sprintlab' hierarchy."""
    _configure_root()
    short = name.split('.')[-1] if name.startswith('services') else name
    return logging.getLogger(f'sprintlab.{short}')


def set_level(level: str) -> None:
    """Change the level of every sprintlab handler at runtime (used by tests)."""
    _configure_root()
    logging.getLogger('sprintlab').setLevel(getattr(logging, level.upper(), logging.INFO))
