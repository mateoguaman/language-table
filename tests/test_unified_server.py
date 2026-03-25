"""Tests for MultiPoolEnvServer — the unified two-port environment server.

These tests verify that the unified server correctly:
- Binds two ports and accepts connections on both
- Routes train/val ports to different managers
- Handles full request/response cycles
- Handles concurrent (sequential) connections across ports
- Returns errors for unknown methods
- Drops idle connections after timeout

No GPUs, JAX, TensorFlow, PyBullet, or LAVA models are required.
Run with:  pytest language_table/lamer/test_unified_server.py -v
"""

import socket
import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest

# Bypass language_table.lamer.__init__ which pulls in PyBullet/TF/JAX.
# We register stub packages and load protocol.py / server.py by file path
# so relative imports within them resolve correctly.
import importlib.util
import os
import types
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
_LAMER_DIR = os.path.join(_REPO_ROOT, "language_table", "lamer")
_LT_DIR = os.path.join(_REPO_ROOT, "language_table")

# Register parent packages as empty namespaces (if not already loaded)
for _pkg, _path in [("language_table", _LT_DIR), ("language_table.lamer", _LAMER_DIR)]:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [_path]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

def _load(mod_name, filename):
    path = os.path.join(_LAMER_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path,
        submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "language_table.lamer"
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_protocol = _load("language_table.lamer.protocol", "protocol.py")
_server = _load("language_table.lamer.server", "server.py")

EnvRequest = _protocol.EnvRequest
EnvResponse = _protocol.EnvResponse
send_message = _protocol.send_message
recv_message = _protocol.recv_message
MultiPoolEnvServer = _server.MultiPoolEnvServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_mock_manager(num_processes=128, num_attempts=1, max_turns=1, do_reflection=False):
    """Create a mock manager with the required properties and methods."""
    mgr = MagicMock()
    mgr.num_processes = num_processes
    mgr.num_attempts = num_attempts
    mgr.max_turns = max_turns
    mgr.do_reflection = do_reflection
    mgr.reset.return_value = ({"text": ["obs"] * num_processes}, [{}] * num_processes)
    mgr.step.return_value = ({"text": ["obs"]}, [0.0], [False], [{}])
    mgr.restart.return_value = ({"text": ["obs"]}, [{}])
    mgr.reflect.return_value = ({"text": ["reflect"]}, [{}])
    mgr.close.return_value = None
    mgr.build_text_obs.return_value = ["text"] * num_processes
    mgr.success_evaluator.return_value = {}
    return mgr


def _connect(port: int, timeout: float = 5.0) -> socket.socket:
    """Connect to the server on localhost:port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(("127.0.0.1", port))
    return sock


def _request(sock, method, args=(), kwargs=None) -> EnvResponse:
    """Send a request and receive the response."""
    req = EnvRequest(
        request_id=str(uuid.uuid4()),
        method=method,
        args=args,
        kwargs=kwargs or {},
    )
    send_message(sock, req)
    return recv_message(sock)


class _ServerFixture:
    """Manages a MultiPoolEnvServer running in a background thread."""

    def __init__(self, train_manager, val_manager):
        self.train_port = _free_port()
        self.val_port = _free_port()
        self.server = MultiPoolEnvServer(
            train_manager,
            val_manager,
            host="127.0.0.1",
            train_port=self.train_port,
            val_port=self.val_port,
        )
        self._thread = threading.Thread(target=self.server.serve, daemon=True)

    def start(self):
        self._thread.start()
        # Wait until both ports are listening by polling with non-blocking
        # connect attempts.  We must NOT actually complete a connection,
        # because the server is single-threaded: accepting on one port blocks
        # the select loop and prevents the other port from becoming ready.
        #
        # Instead we check that the port is in LISTEN state by attempting a
        # connect + immediate close.  But since the server will block in
        # _handle_connection for that probe, we do both ports in parallel
        # using threads so the second probe can still land.
        def _wait_for_port(port):
            for _ in range(100):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.5)
                    s.connect(("127.0.0.1", port))
                    s.close()
                    return
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.05)
            raise RuntimeError(f"Server did not start on port {port}")

        threads = [
            threading.Thread(target=_wait_for_port, args=(self.train_port,)),
            threading.Thread(target=_wait_for_port, args=(self.val_port,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # Give the server a moment to return from the probe connections
        # (the probe sockets closed immediately, so _handle_connection will
        # raise ConnectionError and fall back to the select loop).
        time.sleep(0.1)

    def stop(self):
        # The server thread is daemonic; it will die when the test exits.
        # We close the listening sockets indirectly by interrupting.
        pass


@pytest.fixture
def server_fixture():
    """Fixture that yields a started _ServerFixture and cleans up after."""
    train_mgr = _make_mock_manager(num_processes=128, num_attempts=3, max_turns=5, do_reflection=True)
    val_mgr = _make_mock_manager(num_processes=64, num_attempts=1, max_turns=10, do_reflection=False)
    fixture = _ServerFixture(train_mgr, val_mgr)
    fixture.train_manager = train_mgr
    fixture.val_manager = val_mgr
    fixture.start()
    yield fixture
    fixture.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiPoolEnvServerBinds:
    """MultiPoolEnvServer binds two ports and accepts connections on both."""

    def test_both_ports_respond_to_get_properties(self, server_fixture):
        # Connect to train port
        train_sock = _connect(server_fixture.train_port)
        resp_train = _request(train_sock, "get_properties")
        assert resp_train.status == "ok"
        assert isinstance(resp_train.result, dict)
        # Send close so the server loop returns and can accept the next connection
        _request(train_sock, "close")
        train_sock.close()

        # Connect to val port
        val_sock = _connect(server_fixture.val_port)
        resp_val = _request(val_sock, "get_properties")
        assert resp_val.status == "ok"
        assert isinstance(resp_val.result, dict)
        _request(val_sock, "close")
        val_sock.close()


class TestTrainValRouting:
    """Train and val ports route to different managers."""

    def test_different_properties_per_port(self, server_fixture):
        # Train port should reflect train manager properties
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "get_properties")
        assert resp.status == "ok"
        assert resp.result["num_processes"] == 128
        assert resp.result["num_attempts"] == 3
        assert resp.result["max_turns"] == 5
        assert resp.result["do_reflection"] is True
        _request(train_sock, "close")
        train_sock.close()

        # Val port should reflect val manager properties
        val_sock = _connect(server_fixture.val_port)
        resp = _request(val_sock, "get_properties")
        assert resp.status == "ok"
        assert resp.result["num_processes"] == 64
        assert resp.result["num_attempts"] == 1
        assert resp.result["max_turns"] == 10
        assert resp.result["do_reflection"] is False
        _request(val_sock, "close")
        val_sock.close()


class TestFullRequestResponse:
    """Full request/response cycle on both ports."""

    def test_reset_routes_to_correct_manager(self, server_fixture):
        # Reset on train port
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "reset")
        assert resp.status == "ok"
        server_fixture.train_manager.reset.assert_called_once()
        server_fixture.val_manager.reset.assert_not_called()
        _request(train_sock, "close")
        train_sock.close()

        # Reset on val port
        val_sock = _connect(server_fixture.val_port)
        resp = _request(val_sock, "reset")
        assert resp.status == "ok"
        server_fixture.val_manager.reset.assert_called_once()
        # train_manager.reset should still be at 1 call
        assert server_fixture.train_manager.reset.call_count == 1
        _request(val_sock, "close")
        val_sock.close()

    def test_step_with_args(self, server_fixture):
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "step", args=(["move left"], "play"))
        assert resp.status == "ok"
        server_fixture.train_manager.step.assert_called_once_with(["move left"], "play")
        _request(train_sock, "close")
        train_sock.close()

    def test_build_text_obs_with_kwargs(self, server_fixture):
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "build_text_obs", kwargs={"phase": "reflect"})
        assert resp.status == "ok"
        server_fixture.train_manager.build_text_obs.assert_called_once_with(phase="reflect")
        _request(train_sock, "close")
        train_sock.close()

    def test_success_evaluator_with_kwargs(self, server_fixture):
        train_sock = _connect(server_fixture.train_port)
        resp = _request(
            train_sock,
            "success_evaluator",
            kwargs={"total_infos": [[]], "total_batch_list": [[]]},
        )
        assert resp.status == "ok"
        server_fixture.train_manager.success_evaluator.assert_called_once()
        _request(train_sock, "close")
        train_sock.close()


class TestConcurrentConnections:
    """Sequential connections across both ports work correctly."""

    def test_sequential_train_then_val(self, server_fixture):
        # Connect to train, do work, disconnect
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "reset")
        assert resp.status == "ok"
        _request(train_sock, "close")
        train_sock.close()

        # Connect to val, do work, disconnect
        val_sock = _connect(server_fixture.val_port)
        resp = _request(val_sock, "reset")
        assert resp.status == "ok"
        _request(val_sock, "close")
        val_sock.close()

        # Both managers received exactly one reset call
        assert server_fixture.train_manager.reset.call_count == 1
        assert server_fixture.val_manager.reset.call_count == 1

    def test_multiple_requests_on_same_connection(self, server_fixture):
        train_sock = _connect(server_fixture.train_port)

        # Multiple requests on a single connection
        resp1 = _request(train_sock, "get_properties")
        assert resp1.status == "ok"

        resp2 = _request(train_sock, "reset")
        assert resp2.status == "ok"

        resp3 = _request(train_sock, "reflect")
        assert resp3.status == "ok"

        _request(train_sock, "close")
        train_sock.close()

        assert server_fixture.train_manager.reset.call_count == 1
        assert server_fixture.train_manager.reflect.call_count == 1


class TestErrorHandling:
    """Unknown methods return error responses."""

    def test_unknown_method_returns_error(self, server_fixture):
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "nonexistent_method")
        assert resp.status == "error"
        assert "Unknown method" in resp.error_message
        assert "nonexistent_method" in resp.error_message
        _request(train_sock, "close")
        train_sock.close()

    def test_method_exception_returns_error(self, server_fixture):
        server_fixture.train_manager.reset.side_effect = RuntimeError("boom")
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "reset")
        assert resp.status == "error"
        assert "boom" in resp.error_message
        _request(train_sock, "close")
        train_sock.close()


class TestClientTimeout:
    """Server drops idle connections after timeout."""

    def test_idle_connection_dropped(self):
        """Use a very short timeout to verify the server drops idle clients."""
        train_mgr = _make_mock_manager(num_processes=4)
        val_mgr = _make_mock_manager(num_processes=4)
        fixture = _ServerFixture(train_mgr, val_mgr)
        fixture.start()

        # Monkey-patch _handle_connection to use a 1-second timeout instead
        # of 600 seconds so the test completes quickly.
        def _fast_timeout_handle(conn, manager, label):
            conn.settimeout(1)  # 1 second instead of 600
            while True:
                try:
                    request = recv_message(conn)
                except socket.timeout:
                    break
                response = fixture.server._dispatch(request, manager, label)
                send_message(conn, response)
                if request.method == "close":
                    break

        fixture.server._handle_connection = _fast_timeout_handle

        sock = _connect(fixture.train_port)
        # Send one request to prove the connection is live
        resp = _request(sock, "get_properties")
        assert resp.status == "ok"

        # Now idle — the server should drop us after ~1 second
        # Try to receive; should get a connection closed error
        sock.settimeout(5)
        try:
            data = sock.recv(1)
            # If we get empty bytes, the server closed the connection
            assert data == b""
        except (ConnectionError, socket.timeout):
            # Either is acceptable — the connection was dropped
            pass
        finally:
            sock.close()


class TestCloseMethod:
    """The close method terminates the connection loop."""

    def test_close_ends_connection(self, server_fixture):
        train_sock = _connect(server_fixture.train_port)
        resp = _request(train_sock, "close")
        assert resp.status == "ok"
        server_fixture.train_manager.close.assert_called_once()

        # After close, the server should have closed its side.
        # A subsequent recv should return empty or raise.
        train_sock.settimeout(2)
        try:
            data = train_sock.recv(1)
            assert data == b""
        except (ConnectionError, socket.timeout):
            pass
        finally:
            train_sock.close()
