#!/usr/bin/env python3
"""Verify the public artifact without modifying files or accessing a provider."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def deny_network() -> None:
    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for the offline smoke")

    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deny-network", action="store_true")
    args = parser.parse_args()
    if args.deny_network:
        deny_network()

    from reproducibility.release import verify_release

    for label in verify_release():
        print(f"PASS {label}")
    print("OFFLINE_RELEASE_CHECK=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
