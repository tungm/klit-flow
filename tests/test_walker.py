from pathlib import Path

import pytest

from klit_flow.walker import SourceFile, walk

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mini_app"


def _rel_paths(files: list[SourceFile]) -> set[str]:
    return {f.rel_path for f in files}


def test_discovers_source_files():
    files = walk(FIXTURE_ROOT)
    paths = _rel_paths(files)
    assert "src/MainActivity.kt" in paths
    assert "src/AuthActivity.kt" in paths
    assert "src/Utils.kt" in paths


def test_gitignore_excludes_build_dir():
    files = walk(FIXTURE_ROOT)
    assert "build/output.kt" not in _rel_paths(files)


def test_klit_flowignore_excludes_file():
    files = walk(FIXTURE_ROOT)
    assert "secret.kt" not in _rel_paths(files)


def test_non_source_files_excluded():
    files = walk(FIXTURE_ROOT)
    # README.md has no supported extension
    assert not any(f.rel_path.endswith(".md") for f in files)


def test_language_detection():
    files = walk(FIXTURE_ROOT)
    by_path = {f.rel_path: f for f in files}
    assert by_path["src/MainActivity.kt"].language == "kotlin"
    assert by_path["src/AuthActivity.kt"].language == "kotlin"


def test_returns_sorted_stable_list():
    files = walk(FIXTURE_ROOT)
    rel = [f.rel_path for f in files]
    assert rel == sorted(rel)


def test_max_file_size_skip(tmp_path: Path):
    big = tmp_path / "big.kt"
    big.write_bytes(b"// placeholder\n" * 10)  # 150 bytes
    small = tmp_path / "small.kt"
    small.write_bytes(b"// ok\n")

    files = walk(tmp_path, max_file_size=100)
    paths = _rel_paths(files)
    assert "small.kt" in paths
    assert "big.kt" not in paths


def test_empty_dir_returns_empty_list(tmp_path: Path):
    assert walk(tmp_path) == []


def test_source_file_is_frozen():
    files = walk(FIXTURE_ROOT)
    with pytest.raises((AttributeError, TypeError)):
        files[0].language = "mutated"  # type: ignore[misc]
