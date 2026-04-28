"""
LeRobot policy wrapper for Language Table evaluation.

Implements the same PyPolicy interface as BCJaxPyPolicy, but delegates
inference to a LeRobot policy server over TCP. This allows the eval loop
(running in ltvenv with JAX/PyBullet) to use a LeRobot policy (running
in lerobot_env with PyTorch).

Usage in eval:
    wrapper = LeRobotPolicyClient(host="127.0.0.1", port=50100)
    # ... then use like any PyPolicy:
    policy_step = wrapper.action(time_step, ())
"""

import io
import pickle
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step as ps

# Wire protocol (same as lerobot_policy_server.py and language_table/lamer/protocol.py)
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(_HEADER_FMT, len(payload))
    sock.sendall(header + payload)


def _recv_message(sock: socket.socket) -> Any:
    header = _recv_exact(sock, _HEADER_SIZE)
    (length,) = struct.unpack(_HEADER_FMT, header)
    payload = _recv_exact(sock, length)
    return pickle.loads(payload)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4 * 1024 * 1024))
        if not chunk:
            raise ConnectionError(
                f"Connection closed ({nbytes - remaining}/{nbytes} bytes received)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class PolicyRequest:
    """Mirrors the server-side dataclass."""
    method: str
    rgb: Any = None
    state: Any = None
    instruction: str = ""


@dataclass
class PolicyResponse:
    """Mirrors the server-side dataclass."""
    status: str
    action: Any = None
    error_message: str = ""


class LeRobotPolicyClient(py_policy.PyPolicy):
    """PyPolicy that delegates to a remote LeRobot policy server.

    Designed for use with Language Table eval: takes observations from the
    HistoryWrapper and sends only the most recent frame to the server
    (LeRobot policies handle temporal context internally via action chunks).
    """

    def __init__(self, time_step_spec, action_spec,
                 host: str = "127.0.0.1", port: int = 50100,
                 connect_timeout: float = 60.0):
        super().__init__(time_step_spec, action_spec)
        self._host = host
        self._port = port
        self._sock = None
        self._connect(connect_timeout)

    def _connect(self, timeout: float):
        """Connect to the policy server, retrying until timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.connect((self._host, self._port))
                print(f"Connected to policy server at {self._host}:{self._port}")
                return
            except ConnectionRefusedError:
                self._sock.close()
                time.sleep(1.0)
        raise ConnectionError(
            f"Could not connect to policy server at {self._host}:{self._port} "
            f"within {timeout}s")

    def _action(self, time_step, policy_state=()):
        """Get action from the remote LeRobot policy.

        Expects time_step.observation to contain:
            - 'rgb': image array, possibly with history dim (seq, H, W, 3) or (H, W, 3)
            - 'effector_translation': state array, possibly (seq, 2) or (2,)
            - 'instruction': raw bytes array (the language instruction)
        """
        obs = time_step.observation

        # Extract most recent frame (HistoryWrapper stacks along axis 0)
        rgb = obs['rgb']
        if rgb.ndim == 4:
            rgb = rgb[-1]  # Take last frame from history
        # Ensure uint8
        if rgb.dtype == np.float32:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Extract state (effector position)
        state = obs.get('effector_translation', np.zeros(2, dtype=np.float32))
        if state.ndim == 2:
            state = state[-1]  # Take last from history
        state = state.astype(np.float32)

        # Decode instruction
        instruction_raw = obs.get('instruction', b'')
        if isinstance(instruction_raw, np.ndarray):
            non_zero = instruction_raw[instruction_raw != 0]
            if non_zero.shape[0] > 0:
                instruction = bytes(non_zero.tolist()).decode('utf-8')
            else:
                instruction = ""
        elif isinstance(instruction_raw, bytes):
            instruction = instruction_raw.decode('utf-8')
        else:
            instruction = str(instruction_raw)

        # Send request as plain dict with arrays as nested lists — client
        # (numpy 1.x) and server (numpy 2.x) can't unpickle each other's
        # ndarrays. Lists pickle identically across numpy versions.
        request = {
            "method": "action",
            "rgb": rgb.tolist(),
            "state": state.tolist(),
            "instruction": instruction,
        }
        _send_message(self._sock, request)

        # Receive response
        response = _recv_message(self._sock)
        if response.get("status") != "ok":
            raise RuntimeError(f"Policy server error: {response.get('error_message')}")

        action = np.array(response["action"], dtype=np.float32)
        # Clip to action spec
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)

        return ps.PolicyStep(action=action)

    def reset(self):
        """Signal episode reset to the server (clears action chunk state)."""
        _send_message(self._sock, {"method": "reset"})
        response = _recv_message(self._sock)
        if response.get("status") != "ok":
            raise RuntimeError(f"Policy server error on reset: {response.get('error_message')}")

    def close(self):
        """Close connection to the server."""
        if self._sock:
            try:
                _send_message(self._sock, {"method": "close"})
                _recv_message(self._sock)
            except Exception:
                pass
            self._sock.close()
            self._sock = None

    def __del__(self):
        self.close()
