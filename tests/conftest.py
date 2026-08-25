import os
import sys
import random

import pytest
import numpy as np

#never attempt a real SciBERT/model download during the suite; individual
#tests opt back in explicitly when they exercise the semantic paths
os.environ.setdefault('USE_SCIBERT_NER', '0')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.Interfaces import SigletonMeta


@pytest.fixture(autouse=True)
def _seed_rng():
    random.seed(42)
    np.random.seed(42)
    yield


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Data_Service is a singleton; wipe its instance dict so every test
    starts with a clean store regardless of execution order."""
    SigletonMeta._instances.clear()
    yield
    SigletonMeta._instances.clear()


@pytest.fixture(autouse=True)
def _quiet_logs():
    """Keep service INFO logs out of the test log file; the production
    log stays at its env-configured level outside of tests."""
    from services import log_setup
    log_setup.get_logger('conftest')
    previous = os.environ.get('SPRINTLAB_LOG_LEVEL')
    os.environ['SPRINTLAB_LOG_LEVEL'] = 'WARNING'
    log_setup.set_level('WARNING')
    yield
    if previous is None:
        os.environ.pop('SPRINTLAB_LOG_LEVEL', None)
    else:
        os.environ['SPRINTLAB_LOG_LEVEL'] = previous
    log_setup.set_level(previous or 'INFO')
