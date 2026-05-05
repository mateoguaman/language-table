import atexit, pickle, socket, struct, subprocess, time
import os
import numpy as np

REPO = os.path.expanduser("~/projects/language-table")
CONDA_ENV = "/home/sidhraja/miniconda3/envs/lerobotenv"
use_conda = True
if use_conda:
    LEROBOT_PYTHON = f"{CONDA_ENV}/bin/python"
else:
    LEROBOT_PYTHON = f"{REPO}/lerobotenv/bin/python"
SERVER_SCRIPT  = f"{REPO}/training/eval/lerobot_policy_server.py"

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

def _send(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(_HEADER_FMT, len(payload)) + payload)

def _recv(sock):
    head = b""
    while len(head) < _HEADER_SIZE:
        chunk = sock.recv(_HEADER_SIZE - len(head))
        if not chunk:
            raise ConnectionError("socket closed mid-header")
        head += chunk
    (n,) = struct.unpack(_HEADER_FMT, head)
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-payload")
        buf += chunk
    return pickle.loads(buf)


class SmolVLAPolicy:
    def __init__(self, checkpoint_path,
                 host="127.0.0.1", port=50100,
                 server_log="/tmp/smolvla_interactive.log",
                 ready_timeout=300.0,
                 use_batch=True,
                 seed=0):
        self.host, self.port = host, port
        self.proc, self.sock = None, None
        self.use_batch = use_batch
        self.seed = seed
        self._spawn_server(checkpoint_path, server_log, ready_timeout, seed)
        self._connect()
        atexit.register(self.close)

    def _spawn_server(self, checkpoint, log_path, timeout, seed):
        log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [LEROBOT_PYTHON, "-u", SERVER_SCRIPT,
             "--checkpoint_path", checkpoint,
             "--host", self.host, "--port", str(self.port),
             "--seed", str(seed)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            with open(log_path) as f:
                if "Policy server listening" in f.read():
                    print(f"SmolVLA server up (pid={self.proc.pid}, log={log_path})")
                    return
            if self.proc.poll() is not None:
                with open(log_path) as f:
                    raise RuntimeError(f"server died:\n{f.read()[-2000:]}")
            time.sleep(1.0)
        self.proc.terminate()
        raise RuntimeError(f"server not ready within {timeout}s; see {log_path}")

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def reset(self, num_envs=1, seed=None):
        if seed is not None:
            self.seed = seed
        _send(self.sock, {"method": "reset", "seed": self.seed})
        resp = _recv(self.sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"reset failed: {resp.get('error_message')}")

    def predict(self, goals, obs_list, active_mask):
        if self.use_batch:
            try:
                return self._predict_batch(goals, obs_list, active_mask)
            except RuntimeError as exc:
                if "Unknown method: action_batch" not in str(exc):
                    raise
                self.use_batch = False
                print("SmolVLA server does not support action_batch; falling back to serial actions")
        return self._predict_serial(goals, obs_list, active_mask)

    def _predict_serial(self, goals, obs_list, active_mask):
        actions = []
        for goal, obs, active in zip(goals, obs_list, active_mask):
            if not active:
                actions.append(np.zeros(2, dtype=np.float32))
                continue
            rgb = obs["rgb"]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            state = np.asarray(
                obs.get("effector_translation", np.zeros(2, dtype=np.float32)),
                dtype=np.float32,
            )
            _send(self.sock, {
                "method": "action",
                "rgb": rgb.tolist(),
                "state": state.tolist(),
                "instruction": goal,
            })
            resp = _recv(self.sock)
            # print(f"Action: {resp}")
            if resp.get("status") != "ok":
                raise RuntimeError(f"action failed: {resp.get('error_message')}")
            actions.append(np.asarray(resp["action"], dtype=np.float32))
        return actions

    def _predict_batch(self, goals, obs_list, active_mask):
        active_mask = np.asarray(active_mask, dtype=bool)
        if len(goals) != len(obs_list) or len(goals) != len(active_mask):
            raise ValueError(
                "goals, obs_list, and active_mask must have the same length"
            )

        rgb_batch = []
        state_batch = []
        instructions = []
        for goal, obs, active in zip(goals, obs_list, active_mask):
            instructions.append(goal)
            if not active:
                rgb_batch.append(None)
                state_batch.append(None)
                continue

            rgb = obs["rgb"]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            state = np.asarray(
                obs.get("effector_translation", np.zeros(2, dtype=np.float32)),
                dtype=np.float32,
            )
            rgb_batch.append(rgb.tolist())
            state_batch.append(state.tolist())

        _send(self.sock, {
            "method": "action_batch",
            "rgbs": rgb_batch,
            "states": state_batch,
            "instructions": instructions,
            "active_mask": active_mask.tolist(),
        })
        resp = _recv(self.sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"action_batch failed: {resp.get('error_message')}")

        raw_actions = resp.get("actions")
        if raw_actions is None or len(raw_actions) != len(goals):
            raise RuntimeError(
                f"action_batch returned invalid actions: {raw_actions!r}"
            )

        actions = []
        for raw_action, active in zip(raw_actions, active_mask):
            if not active:
                actions.append(np.zeros(2, dtype=np.float32))
            else:
                actions.append(np.asarray(raw_action, dtype=np.float32))
        return actions

    def close(self):
        if self.sock is not None:
            try:
                _send(self.sock, {"method": "close"})
                _recv(self.sock)
            except Exception:
                pass
            try: self.sock.close()
            except Exception: pass
            self.sock = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired: self.proc.kill()
            self.proc = None

    def __del__(self):
        self.close()