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

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 n_action_steps: int = 1):
        import json
        import os
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        print(f"Loading policy from: {checkpoint_path}")
        # Resolve HF repo IDs to a local snapshot dir so the rest of the loader
        # (which reads files directly) works the same as for local checkpoints.
        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download
            print(f"Not a local dir; resolving '{checkpoint_path}' via HF Hub.")
            checkpoint_path = snapshot_download(repo_id=checkpoint_path)
            print(f"Snapshot at: {checkpoint_path}")

        # Read policy type from config to dispatch the right concrete class
        # (PreTrainedPolicy itself is abstract).
        with open(f"{checkpoint_path}/config.json") as f:
            policy_type = json.load(f)["type"]
        policy_cls = get_policy_class(policy_type)
        self.policy = policy_cls.from_pretrained(checkpoint_path)
        self.policy.to(device)
        self.policy.eval()
        self.device = device

        # Load the saved preprocessor/postprocessor pipelines. These handle
        # rename (rgb -> camera1 for SmolVLA), tokenization, normalization,
        # and device placement. Override device to match this server's GPU.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint_path,
            preprocessor_overrides={
                "device_processor": {"device": device},
            },
        )
        self.n_action_steps = max(1, n_action_steps)
        print(f"Policy loaded on {device}. Type: {type(self.policy).__name__}")
        print(f"Preprocessor steps: "
              f"{[type(s).__name__ for s in self.preprocessor.steps]}")
        print(f"n_action_steps: {self.n_action_steps}")

    def get_actions(self, rgb, state, instruction: str) -> np.ndarray:
        """Return n_action_steps consecutive actions from the policy's chunk buffer.

        The first call triggers a full forward pass (filling the internal
        action-chunk queue). Subsequent calls within the same chunk drain the
        queue without re-running inference. A new forward pass runs
        automatically when the queue is exhausted.

        Args:
            rgb: uint8 image as nested list (H, W, 3) or ndarray
            state: float32 end-effector position as list (2,) or ndarray
            instruction: language instruction text

        Returns:
            float32 array of shape (n_action_steps, 2)
        """
        rgb = np.asarray(rgb, dtype=np.uint8)
        state = np.asarray(state, dtype=np.float32)

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()

        batch = {
            "observation.images.rgb": img_tensor,
            "observation.state": state_tensor,
            "task": instruction,
        }

        actions = []
        with torch.no_grad():
            for _ in range(self.n_action_steps):
                b = self.preprocessor(batch)
                action = self.policy.select_action(b)
                action = self.postprocessor(action)
                actions.append(action.cpu().numpy().flatten()[:2])

        return np.stack(actions, axis=0)  # (n_action_steps, 2)

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
        """Handle a single client connection. Wire format is a plain dict so
        that client and server don't need to share the dataclass module
        (the client lives in ltvenv with tf_agents; the server lives in
        lerobotenv without it)."""
        while True:
            request = recv_message(conn)
            method = request.get("method")

            if method == "close":
                send_message(conn, {"status": "ok"})
                break
            elif method == "reset":
                self.reset()
                send_message(conn, {"status": "ok"})
            elif method == "action":
                try:
                    # Returns (n_action_steps, 2); sent as nested list so
                    # numpy version mismatches between envs don't matter.
                    actions = self.get_actions(
                        request["rgb"], request["state"], request["instruction"])
                    send_message(conn, {
                        "status": "ok", "actions": actions.tolist()})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    send_message(conn, {
                        "status": "error", "error_message": str(e)})
            else:
                send_message(conn, {
                    "status": "error",
                    "error_message": f"Unknown method: {method}"})


def main():
    parser = argparse.ArgumentParser(description="LeRobot policy server for Language Table eval")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to LeRobot pretrained_model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50100)
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (cuda or cpu)")
    parser.add_argument("--n_action_steps", type=int, default=1,
                        help="Number of consecutive actions to return per request "
                             "(must be <= policy action_chunk_size).")
    args = parser.parse_args()

    server = LeRobotPolicyServer(
        args.checkpoint_path, device=args.device,
        n_action_steps=args.n_action_steps,
    )
    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()