#!/usr/bin/env python3
"""
Connection test for Language Table <-> LaMer remote env server.

Tests TCP connectivity, protocol round-trips, and the full meta-RL loop
(reset → step → restart → reflect) without needing a GPU or LLM model.
Works locally and on SLURM — run from ANY machine that can reach the server.

Must be run from the language-table venv (needs language_table.lamer.protocol
for pickle-compatible dataclasses).

Usage
-----
# Basic connectivity check (just verifies TCP + get_properties)
python -m language_table.lamer.test_connection --host localhost --port 50051

# Full protocol test (reset, step, restart, reflect, success_evaluator)
python -m language_table.lamer.test_connection --host localhost --port 50051 --full

# Test both train + val servers
python -m language_table.lamer.test_connection \
    --host localhost --port 50051 --val_port 50052 --full

# SLURM: test cross-node connectivity (server runs on compute node)
python -m language_table.lamer.test_connection \
    --host compute-node-hostname --port 50051 --full
"""

import argparse
import socket
import sys
import time

import numpy as np

from language_table.lamer.protocol import (
    EnvRequest, EnvResponse, send_message, recv_message,
)


def _call(sock, method, *args, **kwargs):
    req = EnvRequest(request_id=str(time.monotonic()), method=method, args=args, kwargs=kwargs)
    send_message(sock, req)
    resp = recv_message(sock)
    if resp.status == "error":
        raise RuntimeError(f"Server error in '{method}':\n{resp.error_message}")
    return resp.result


# ---- Test functions ----

def test_tcp_connect(host, port, timeout=10):
    """Test 1: Can we open a TCP connection?"""
    print(f"\n  [1/6] TCP connect to {host}:{port} ...", end=" ", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        print("OK")
        return sock
    except (ConnectionRefusedError, OSError, socket.timeout) as e:
        print(f"FAILED: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    - Is the server running? Start it with:")
        print(f"      ltvenv/bin/python -m language_table.lamer.server_main --host 0.0.0.0 --port {port}")
        print(f"    - On SLURM, is the server on the same node or is the port accessible?")
        print(f"      Check with: ssh {host} 'ss -tlnp | grep {port}'")
        print(f"    - Firewall? Try: nc -zv {host} {port}")
        return None


def test_get_properties(sock):
    """Test 2: Protocol works — get_properties."""
    print("  [2/6] get_properties ...", end=" ", flush=True)
    props = _call(sock, "get_properties")
    print(f"OK (num_processes={props['num_processes']}, "
          f"num_attempts={props['num_attempts']}, "
          f"max_turns={props['max_turns']}, "
          f"do_reflection={props['do_reflection']})")
    return props


def test_reset(sock, num_processes):
    """Test 3: reset() returns valid obs."""
    print("  [3/6] reset ...", end=" ", flush=True)
    t0 = time.perf_counter()
    obs, infos = _call(sock, "reset")
    elapsed = time.perf_counter() - t0
    text_obs = obs["text"]
    assert len(text_obs) == num_processes, f"Expected {num_processes} text obs, got {len(text_obs)}"
    assert len(text_obs[0]) > 0, "Empty text observation"
    print(f"OK ({elapsed:.2f}s, {len(text_obs[0])} chars in first obs)")
    return obs


def test_step(sock, num_processes):
    """Test 4: step() with dummy goal strings."""
    print("  [4/6] step (play phase) ...", end=" ", flush=True)
    goals = ["push the red star to the blue cube"] * num_processes
    t0 = time.perf_counter()
    obs, rewards, dones, infos = _call(sock, "step", goals, phase="play")
    elapsed = time.perf_counter() - t0
    rewards = np.asarray(rewards)
    dones = np.asarray(dones)
    assert rewards.shape == (num_processes,), f"Rewards shape {rewards.shape}"
    assert dones.shape == (num_processes,), f"Dones shape {dones.shape}"
    print(f"OK ({elapsed:.2f}s, rewards={rewards.tolist()}, dones={dones.tolist()})")
    return obs, rewards, dones


def test_restart(sock, num_processes, pre_reset_obs=None):
    """Test 5: restart() returns valid obs (and matches reset if available)."""
    print("  [5/6] restart (meta-RL) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    obs, infos = _call(sock, "restart")
    elapsed = time.perf_counter() - t0
    text_obs = obs["text"]
    assert len(text_obs) == num_processes
    print(f"OK ({elapsed:.2f}s)")


def test_reflect_cycle(sock, num_processes):
    """Test 6: Full reflect → step(reflect) → restart cycle."""
    print("  [6/6] reflect cycle ...", end=" ", flush=True)

    # reflect
    obs, infos = _call(sock, "reflect")
    assert len(obs["text"]) == num_processes

    # step with reflect phase
    reflections = ["I should try a different approach"] * num_processes
    obs, rewards, dones, infos = _call(sock, "step", reflections, phase="reflect")
    rewards = np.asarray(rewards)
    assert rewards.shape == (num_processes,)

    # restart
    obs, infos = _call(sock, "restart")
    assert len(obs["text"]) == num_processes

    print("OK (reflect → step(reflect) → restart)")


def test_latency(sock, num_processes, n_trials=10):
    """Bonus: measure round-trip latency."""
    print(f"\n  Latency benchmark ({n_trials} trials):")
    goals = ["push the red star to the blue cube"] * num_processes

    for op_name, op_fn in [
        ("reset", lambda: _call(sock, "reset")),
        ("step", lambda: _call(sock, "step", goals, phase="play")),
        ("restart", lambda: _call(sock, "restart")),
    ]:
        times = []
        for _ in range(n_trials):
            if op_name == "step":
                _call(sock, "reset")  # ensure valid state
            t0 = time.perf_counter()
            op_fn()
            times.append(time.perf_counter() - t0)
        times = np.array(times) * 1000
        print(f"    {op_name:<10}: p50={np.median(times):.1f}ms  "
              f"p95={np.percentile(times, 95):.1f}ms  "
              f"mean={np.mean(times):.1f}ms")


def run_tests(host, port, full=False, latency=False, timeout=10):
    """Run the test suite against a single server."""
    print(f"\n{'='*60}")
    print(f"Testing server at {host}:{port}")
    print(f"{'='*60}")

    sock = test_tcp_connect(host, port, timeout=timeout)
    if sock is None:
        return False

    try:
        props = test_get_properties(sock)
        num_processes = props["num_processes"]

        if full:
            reset_obs = test_reset(sock, num_processes)
            test_step(sock, num_processes)
            test_restart(sock, num_processes, pre_reset_obs=reset_obs)
            test_reflect_cycle(sock, num_processes)

            if latency:
                test_latency(sock, num_processes)

        print(f"\n  All tests PASSED for {host}:{port}")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            _call(sock, "close")
        except Exception:
            pass
        sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test Language Table env server connectivity and protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", type=str, default="localhost",
                        help="Server hostname or IP")
    parser.add_argument("--port", type=int, default=50051,
                        help="Training server port")
    parser.add_argument("--val_port", type=int, default=None,
                        help="Validation server port (optional, tests both)")
    parser.add_argument("--full", action="store_true",
                        help="Run full protocol tests (not just connectivity)")
    parser.add_argument("--latency", action="store_true",
                        help="Also run latency benchmark (implies --full)")
    parser.add_argument("--timeout", type=float, default=30,
                        help="TCP connection timeout in seconds")
    args = parser.parse_args()

    if args.latency:
        args.full = True

    all_passed = True
    all_passed &= run_tests(args.host, args.port, full=args.full, latency=args.latency, timeout=args.timeout)

    if args.val_port is not None:
        all_passed &= run_tests(args.host, args.val_port, full=args.full, latency=args.latency, timeout=args.timeout)

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL SERVERS OK")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
