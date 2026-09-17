"""Smoke tests: prove the scaffolding and CI pipeline work."""


def test_ci_is_wired() -> None:
    """Trivial passing test so CI goes green."""
    assert 1 + 1 == 2
