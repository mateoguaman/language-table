"""
LeRobot policy server for Language Table evaluation.

Loads a LeRobot checkpoint and serves actions over TCP, using the same
length-prefixed pickle wire format as language_table/lamer/protocol.py.

Run this in the lerobot_env (or any env with torch + lerobot installed):
    python training/eval/lerobot_policy_server.py \
        --checkpoint_path outputs/smolvla_expert_oracle/checkpoints/last/pretrained_model \
        --port 50100

The run_eval.py script (in ltvenv) connects as a client.
"""

import argparse
import io
import pickle
import socket
import struct
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Wire protocol (same as language_table/lamer/protocol.py)
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(_HEADER_FMT, len(payload))
    sock.sendall(header + payload)


def recv_message(sock: socket.socket) -> Any:
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
    """Client -> Server: observation for action inference."""
    method: str  # "action" | "reset" | "close"
    rgb: Any = None          # np.ndarray uint8 (H, W, 3)
    state: Any = None        # np.ndarray float32 (2,)
    instruction: str = ""    # language instruction text


@dataclass
class PolicyResponse:
    """Server -> Client: action result."""
    status: str  # "ok" | "error"
    action: Any = None       # np.ndarray float32 (2,)
    error_message: str = ""


class LeRobotPolicyServer:
    """Serves a LeRobot policy over TCP."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from lerobot.policies.pretrained import PreTrainedPolicy

        print(f"Loading policy from: {checkpoint_path}")
        self.policy = PreTrainedPolicy.from_pretrained(checkpoint_path)
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        print(f"Policy loaded on {device}. Type: {type(self.policy).__name__}")

    def get_action(self, rgb: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        """Run inference on a single observation.

        Args:
            rgb: uint8 image (H, W, 3)
            state: float32 end-effector position (2,)
            instruction: language instruction text

        Returns:
            float32 action (2,)
        """
        # Convert to LeRobot batch format
        # Image: (H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()

        batch = {
            "observation.images.rgb": img_tensor.to(self.device),
            "observation.state": state_tensor.to(self.device),
            "task": instruction,
        }

        with torch.no_grad():
            action = self.policy.select_action(batch)

        return action.cpu().numpy().flatten()[:2]  # Ensure 2D action

    def reset(self):
        """Reset policy state (for policies with internal state like action chunks)."""
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def serve(self, host: str = "127.0.0.1", port: int = 50100):
        """Start the TCP server and handle requests."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Policy server listening on {host}:{port}")

        while True:
            conn, addr = server.accept()
            print(f"Client connected from {addr}")
            try:
                self._handle_client(conn)
            except (ConnectionError, EOFError) as e:
                print(f"Client disconnected: {e}")
            finally:
                conn.close()

    def _handle_client(self, conn: socket.socket):
        """Handle a single client connection."""
        while True:
            request = recv_message(conn)

            if request.method == "close":
                send_message(conn, PolicyResponse(status="ok"))
                break
            elif request.method == "reset":
                self.reset()
                send_message(conn, PolicyResponse(status="ok"))
            elif request.method == "action":
                try:
                    action = self.get_action(request.rgb, request.state, request.instruction)
                    send_message(conn, PolicyResponse(status="ok", action=action))
                except Exception as e:
                    send_message(conn, PolicyResponse(
                        status="error", error_message=str(e)))
            else:
                send_message(conn, PolicyResponse(
                    status="error",
                    error_message=f"Unknown method: {request.method}"))


def main():
    parser = argparse.ArgumentParser(description="LeRobot policy server for Language Table eval")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to LeRobot pretrained_model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50100)
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (cuda or cpu)")
    args = parser.parse_args()

    server = LeRobotPolicyServer(args.checkpoint_path, device=args.device)
    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
