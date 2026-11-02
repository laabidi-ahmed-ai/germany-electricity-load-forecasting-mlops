"""The lock file (requirements/constraints.txt) must match what the project declares.

Every job installs with `-c requirements/constraints.txt`, so a dependency added to
pyproject.toml without re-running `make lock`, or a bound changed so that the pinned
version no longer satisfies it, would make the jobs install something untested (or fail
to install). These checks are offline: they read the files, they do not resolve anything.
"""

from __future__ import annotations

import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from config.settings import PROJECT_ROOT

LOCK = PROJECT_ROOT / "requirements" / "constraints.txt"


def locked_versions() -> dict[str, list[Version]]:
    """Package -> pinned version(s); a package may have one pin per Python version."""
    pins: dict[str, list[Version]] = {}
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        spec = line.split(";", 1)[0].strip()
        assert "==" in spec, f"not an exact pin: {raw!r}"
        name, version = spec.split("==", 1)
        pins.setdefault(canonicalize_name(name), []).append(Version(version.strip()))
    return pins


def declared() -> list[str]:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    reqs = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        reqs.extend(extra)
    ingest = (PROJECT_ROOT / "requirements" / "ingest.txt").read_text(encoding="utf-8")
    reqs.extend(
        line.strip()
        for line in ingest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return reqs


def test_lock_contains_only_exact_pins() -> None:
    assert locked_versions()  # parses, and every entry is `name==version`


@pytest.mark.parametrize("requirement", declared())
def test_every_declared_dependency_is_locked_within_its_bounds(requirement: str) -> None:
    req = Requirement(requirement)
    pins = locked_versions().get(canonicalize_name(req.name))
    assert pins, f"{req.name} is declared but not in the lock file - run `make lock`"
    for version in pins:
        assert req.specifier.contains(version, prereleases=True), (
            f"locked {req.name}=={version} violates '{requirement}' - run `make lock`"
        )


def test_lock_keeps_the_known_breaking_majors_out() -> None:
    """The two caps from the October incidents (README §16.2) hold in the lock too."""
    pins = locked_versions()
    assert all(v < Version("3") for v in pins["pandas"])
    assert all(v < Version("2.5") for v in pins["numpy"])
