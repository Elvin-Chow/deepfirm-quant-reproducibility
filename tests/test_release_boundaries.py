from __future__ import annotations

import gzip
from pathlib import Path

from reproducibility.release import ROOT


def _public_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def test_no_provider_price_or_local_cache_files_are_released() -> None:
    forbidden_suffixes = {".parquet", ".feather", ".sqlite", ".sqlite3", ".db"}
    assert not [path for path in _public_files() if path.suffix.lower() in forbidden_suffixes]
    assert not [path for path in _public_files() if "cache" in path.parts or "data" in path.parts]


def test_no_absolute_local_machine_path_is_embedded_in_text_outputs() -> None:
    local_prefixes = ["/" + "Users/", "/" + "home/"]
    offenders = []
    for path in _public_files():
        if path.suffix.lower() in {".png", ".pdf"}:
            continue
        try:
            if path.suffix == ".gz":
                text = gzip.open(path, "rt", encoding="utf-8").read()
            else:
                text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(prefix in text for prefix in local_prefixes):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders


def test_public_execution_surface_contains_no_live_final_run_entry() -> None:
    scripts = {path.name for path in (ROOT / "scripts").glob("*.py")}
    assert scripts == {
        "__init__.py",
        "generate_figures.py",
        "generate_manifest.py",
        "generate_tables.py",
        "verify_release.py",
    }
