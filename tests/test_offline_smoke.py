from __future__ import annotations

import socket

import pytest

from reproducibility.release import verify_release
from scripts.verify_release import deny_network


def test_release_checks_pass_without_provider_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network access was attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    assert verify_release() == [
        "required paths",
        "configuration boundary",
        "artifact hashes and schemas",
        "frozen result schemas",
        "frozen expected values",
        "result manifest",
    ]


def test_socket_guard_fails_closed() -> None:
    original_create_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    try:
        deny_network()
        with pytest.raises(RuntimeError, match="network access is disabled"):
            socket.create_connection(("127.0.0.1", 9))
    finally:
        socket.create_connection = original_create_connection
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
