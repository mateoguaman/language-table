"""
Environment server — wraps LanguageTableEnvironmentManager and exposes
it over TCP using the length-prefixed pickle protocol.

Usage:
    server = EnvServer(env_manager, host="0.0.0.0", port=50051)
    server.serve()          # blocks forever

Copied from LaMer/agent_system/environments/remote/server.py with
local imports adjusted.
"""

import logging
import socket
import time
import traceback

from .protocol import EnvRequest, EnvResponse, send_message, recv_message

logger = logging.getLogger(__name__)

# Methods the client is allowed to call remotely.
_ALLOWED_METHODS = frozenset({
    "reset",
    "step",
    "restart",
    "reflect",
    "success_evaluator",
    "close",
    "build_text_obs",
})

_TIMED_METHODS = frozenset({"reset", "step", "restart", "reflect"})


class EnvServer:
    """Single-client, synchronous environment server."""

    def __init__(self, env_manager, host: str = "0.0.0.0", port: int = 50051):
        self.env = env_manager
        self.host = host
        self.port = port

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serve(self):
        """Accept connections and handle requests until ``close`` is called
        or the process is interrupted."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        logger.info("EnvServer listening on %s:%s", self.host, self.port)

        try:
            while True:
                logger.debug("Waiting for client connection …")
                conn, addr = srv.accept()
                logger.info("Client connected from %s", addr)
                try:
                    self._handle_connection(conn)
                except ConnectionError as exc:
                    logger.debug("Client disconnected: %s", exc)
                finally:
                    conn.close()
                    logger.debug("Connection from %s closed", addr)
        except KeyboardInterrupt:
            logger.info("Server interrupted — shutting down")
        finally:
            srv.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_connection(self, conn: socket.socket):
        """Process requests from a single client until it disconnects or
        sends a ``close`` request."""
        # detect dead clients instead of blocking forever
        # keep this at 600 to ensure initial evaluation doesn't kill the server
        conn.settimeout(600)
        while True:
            try:
                request: EnvRequest = recv_message(conn)
            except socket.timeout:
                logger.warning("Client idle for 600s — dropping connection")
                break
            response = self._dispatch(request)
            send_message(conn, response)
            if request.method == "close":
                break

    def _dispatch(self, request: EnvRequest) -> EnvResponse:
        """Route a request to the wrapped environment manager."""
        try:
            if request.method == "get_properties":
                return self._get_properties(request)

            if request.method not in _ALLOWED_METHODS:
                return EnvResponse(
                    request_id=request.request_id,
                    status="error",
                    error_message=f"Unknown method: {request.method}",
                )

            method = getattr(self.env, request.method)
            t0 = time.monotonic()
            result = method(*request.args, **request.kwargs)
            elapsed = time.monotonic() - t0
            if request.method in _TIMED_METHODS:
                logger.info("%s completed in %.3fs", request.method, elapsed)
            return EnvResponse(
                request_id=request.request_id,
                status="ok",
                result=result,
            )
        except Exception:
            return EnvResponse(
                request_id=request.request_id,
                status="error",
                error_message=traceback.format_exc(),
            )

    def _get_properties(self, request: EnvRequest) -> EnvResponse:
        """Return read-only properties that the client caches."""
        props = {}
        for name in ("num_attempts", "num_processes", "max_turns",
                      "do_reflection"):
            props[name] = getattr(self.env, name, None)
        return EnvResponse(
            request_id=request.request_id,
            status="ok",
            result=props,
        )
