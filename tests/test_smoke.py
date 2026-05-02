"""Smoke test so CI has something to run before v0.1 lands real tests."""

import sarcolit


def test_version_is_set() -> None:
    assert sarcolit.__version__ == "0.0.0"
