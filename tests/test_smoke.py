"""Phase 0 smoke tests: prove the scaffolding and CI pipeline work."""


def test_ci_is_wired() -> None:
    """Trivial passing test so CI goes green (README §15, Phase 0)."""
    assert 1 + 1 == 2
