from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # plotting tests must not open windows

TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def tutorials_dir():
    return TESTS_DIR.parent / "tutorials"


@pytest.fixture(scope="session")
def data_dir():
    return TESTS_DIR / "data"
