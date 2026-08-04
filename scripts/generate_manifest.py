#!/usr/bin/env python3
"""Generate or check the SHA-256 manifest for released result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.crisis_warning_artifact_hash import sha256_file


def manifest_text() -> str:
    rows = []
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file() or path.name == "manifest.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "results/manifest.sha256"
    expected = manifest_text()
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            print("results/manifest.sha256 is stale", file=sys.stderr)
            return 1
        print("results/manifest.sha256 is current")
        return 0
    path.write_text(expected, encoding="utf-8")
    print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
