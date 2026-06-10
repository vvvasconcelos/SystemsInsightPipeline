from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tutorials_dir():
    return Path(__file__).resolve().parent.parent / "tutorials"
