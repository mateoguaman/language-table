import atexit, pickle, socket, struct, subprocess, time
import os
import numpy as np


tillicum = os.environ.get("TILLICUM", False)

if tillicum:
    REPO = "/gpfs/projects/stf/sidhraja/projects/language-table"
    CONDA_ENV = "/gpfs/projects/stf/sidhraja/.conda/envs/lerobotenv"
else:
    REPO = os.path.expanduser("~/projects/language-table")
    CONDA_ENV = "/home/sidhraja/miniconda3/envs/lerobotenv"
use_conda = True
if use_conda:
    LEROBOT_PYTHON = f"{CONDA_ENV}/bin/python"
else:
    LEROBOT_PYTHON = f"{REPO}/lerobotenv/bin/python"
SERVER_SCRIPT  = f"{REPO}/language_table/policies/smolvla_server.py"

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
    """SmolVLA policy client.

    The manager (LanguageTableEnvironmentManager) owns action chunking via its
    per-env FIFO.  This class exposes ``predict_chunk`` which issues one TCP
    request per active env and returns the server's full ``(K_server, 2)``
    array.  The manager trims to its own ``chunk_size`` and stores the rest.

    ``predict`` is kept as a thin back-compat wrapper for standalone test
    scripts (e.g. test_lava_standalone.py) that call it directly.
    """

    def __init__(self, checkpoint_path,
                 host="127.0.0.1", port=50100,
                 n_action_steps=1,
                 server_log=None,
                 ready_timeout=300.0):
        self.host, self.port = host, port
        self.n_action_steps = max(1, n_action_steps)
        self.proc, self.sock = None, None
        if server_log is None:
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            server_log = f"/tmp/smolvla_interactive_{job_id}_{os.getpid()}_{port}.log"
        self._spawn_server(checkpoint_path, n_action_steps, server_log, ready_timeout)
        self._connect()
        atexit.register(self.close)

    def _spawn_server(self, checkpoint, n_action_steps, log_path, timeout):
        log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [LEROBOT_PYTHON, "-u", SERVER_SCRIPT,
             "--checkpoint_path", checkpoint,
             "--host", self.host, "--port", str(self.port),
             "--n_action_steps", str(n_action_steps)],
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
        deadline = time.time() + 30.0
        last_error = None
        while time.time() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((self.host, self.port))
                self.sock = sock
                return
            except OSError as exc:
                last_error = exc
                sock.close()
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"SmolVLA server exited before connect: {last_error}"
                    ) from exc
                time.sleep(0.5)
        raise ConnectionError(
            f"Could not connect to SmolVLA server at {self.host}:{self.port}"
        ) from last_error

    def reset(self, num_envs=1):
        _send(self.sock, {"method": "reset"})
        resp = _recv(self.sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"reset failed: {resp.get('error_message')}")

    def _obs_to_wire(self, obs):
        """Extract RGB and state from an obs dict, ready for the wire."""
        rgb = obs["rgb"]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        state = np.asarray(
            obs.get("effector_translation", np.zeros(2, dtype=np.float32)),
            dtype=np.float32,
        )
        return rgb.tolist(), state.tolist()

    def predict_chunk(self, goals, obs_list, active_mask):
        """Return the server's full action chunk per active env.

        For each active env issues one TCP ``action`` request and returns the
        server's ``(K_server, 2)`` array.  Inactive envs get a ``(1, 2)``
        zero array.

        The caller (LanguageTableEnvironmentManager) owns chunking: it asserts
        ``K_server >= chunk_size`` and keeps only the first ``chunk_size``
        actions per env.

        Returns
        -------
        List[np.ndarray]
            One ``(K_i, 2)`` float32 array per env.  ``K_i == K_server`` for
            active envs, ``K_i == 1`` (zeros) for inactive envs.
        """
        active_indices = [i for i, a in enumerate(active_mask) if a]
        chunks = [np.zeros((1, 2), dtype=np.float32)] * len(goals)
        if not active_indices:
            return chunks

        rgb_batch, state_batch, instructions = [], [], []
        for i in active_indices:
            rgb, state = self._obs_to_wire(obs_list[i])
            rgb_batch.append(rgb)
            state_batch.append(state)
            instructions.append(goals[i])

        _send(self.sock, {
            "method": "action_batch",
            "rgb_batch": rgb_batch,
            "state_batch": state_batch,
            "instructions": instructions,
        })
        resp = _recv(self.sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"action_batch failed: {resp.get('error_message')}")
        actions_batch = np.asarray(resp["actions_batch"], dtype=np.float32)  # (N_active, K, 2)
        for j, i in enumerate(active_indices):
            chunks[i] = actions_batch[j]
        return chunks

    def predict(self, goals, obs_list, active_mask):
        """Back-compat: return one action per env.

        Calls ``predict_chunk`` and returns the first action from each chunk.
        Standalone test scripts that call ``predict`` directly continue to work.
        """
        chunks = self.predict_chunk(goals, obs_list, active_mask)
        return [c[0] for c in chunks]

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
