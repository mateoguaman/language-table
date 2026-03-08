"""
Standalone test/demo script for the Language Table LaMer integration.

Usage:
    ltvenv/bin/python -m language_table.lamer.test_standalone \
        --num_envs 4 --num_steps 50 --output_dir "${LT_RENDER_DIR:-/tmp/lt_renders}"

Tests:
    1. Single env smoke test: 1 env, 10 random steps, save render
    2. Parallel test: N envs, random actions, save grid images + video
    3. Scaling sweep (--scaling_sweep): 1/2/4/8 envs, report throughput
    4. Server round-trip (--test_server): start server subprocess, connect, reset+step
    5. State-to-text test: verify text conversion format
"""

import argparse
import os
import subprocess
import sys
import time
import socket

import cv2
import numpy as np


def tile_images(images, ncols=4):
    """Tile a list of RGB images into a grid."""
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    h, w = images[0].shape[:2]
    grid = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return grid


def test_single_env(output_dir):
    """Test 1: Single env smoke test."""
    print("\n=== Test 1: Single env smoke test ===")
    from language_table.lamer.envs import LanguageTableMultiProcessEnv

    envs = LanguageTableMultiProcessEnv(num_envs=1, block_mode="BLOCK_4", seed=42)
    obs_list, infos = envs.reset()
    print(f"  Reset complete. Obs keys: {sorted(obs_list[0].keys())}")

    for step_i in range(10):
        action = np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
        obs_list, rewards, dones, infos = envs.step([action])
        print(f"  Step {step_i}: reward={rewards[0]:.4f}, done={dones[0]}")

    renders = envs.render()
    img = renders[0]
    path = os.path.join(output_dir, "test1_single_env.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  Saved render to {path} (shape={img.shape})")

    envs.close()
    print("  PASSED")


def test_parallel(num_envs, num_steps, output_dir):
    """Test 2: Parallel envs with grid rendering + video."""
    print(f"\n=== Test 2: Parallel test ({num_envs} envs, {num_steps} steps) ===")
    from language_table.lamer.envs import LanguageTableMultiProcessEnv

    envs = LanguageTableMultiProcessEnv(
        num_envs=num_envs, block_mode="BLOCK_4", seed=0
    )
    envs.reset()

    frames = []
    for step_i in range(num_steps):
        actions = [
            np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
            for _ in range(envs.num_processes)
        ]
        envs.step(actions)

        if step_i % 10 == 0 or step_i == num_steps - 1:
            renders = envs.render()
            grid = tile_images(renders)
            frames.append(grid)

            path = os.path.join(output_dir, f"test2_step{step_i:04d}.png")
            cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # Write video
    if frames:
        h, w = frames[0].shape[:2]
        video_path = os.path.join(output_dir, "test2_parallel.mp4")
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h)
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved to {video_path} ({len(frames)} frames)")

    envs.close()
    print("  PASSED")


def test_scaling_sweep(output_dir, num_cpus_list=None, timeout=10):
    """Test 3: Scaling sweep — measure throughput at different env counts and num_cpus."""
    print("\n=== Test 3: Scaling sweep ===")
    import ray
    from language_table.lamer.envs import LanguageTableMultiProcessEnv

    if num_cpus_list is None:
        num_cpus_list = [0.1]

    num_steps = 20
    env_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # results[(nc, n)] = (steps_per_sec, ms_per_batch, elapsed) or None on timeout
    results = {}

    for nc in num_cpus_list:
        print(f"\n  --- num_cpus={nc} ---")
        for n in env_counts:
            envs = LanguageTableMultiProcessEnv(
                num_envs=n, block_mode="BLOCK_4", seed=0,
                num_cpus=nc, timeout=timeout,
            )
            try:
                envs.reset()

                t0 = time.time()
                for _ in range(num_steps):
                    actions = [
                        np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
                        for _ in range(envs.num_processes)
                    ]
                    envs.step(actions)
                elapsed = time.time() - t0

                total_steps = n * num_steps
                sps = total_steps / elapsed
                ms_batch = elapsed / num_steps * 1000
                results[(nc, n)] = (sps, ms_batch, elapsed)
                print(
                    f"  {n:>4d} envs: {elapsed:.2f}s total, "
                    f"{sps:.1f} steps/s, "
                    f"{ms_batch:.1f} ms/batch-step"
                )
            except ray.exceptions.GetTimeoutError:
                results[(nc, n)] = None
                print(f"  {n:>4d} envs: TIMEOUT")
            finally:
                envs.close()

    # ── Summary table ──
    col_w = 12
    header = "num_cpus".rjust(col_w) + "".join(str(n).rjust(col_w) for n in env_counts)
    sep = "-" * len(header)

    print(f"\n  {'=' * len(header)}")
    print(f"  Results table — steps/s (higher is better)")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for nc in num_cpus_list:
        row = str(nc).rjust(col_w)
        for n in env_counts:
            r = results.get((nc, n))
            row += ("TIMEOUT" if r is None else f"{r[0]:.0f}").rjust(col_w)
        print(f"  {row}")
    print(f"  {sep}")

    print(f"\n  Results table — ms/batch-step (lower is better)")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for nc in num_cpus_list:
        row = str(nc).rjust(col_w)
        for n in env_counts:
            r = results.get((nc, n))
            row += ("TIMEOUT" if r is None else f"{r[1]:.1f}").rjust(col_w)
        print(f"  {row}")
    print(f"  {sep}")

    # ── Optimal setting ──
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        best_key = max(valid, key=lambda k: valid[k][0])
        best_nc, best_n = best_key
        best_sps, best_ms, best_elapsed = valid[best_key]
        print(
            f"\n  >>> Best throughput: {best_sps:.1f} steps/s "
            f"at num_cpus={best_nc}, num_envs={best_n} "
            f"({best_ms:.1f} ms/batch-step, {best_elapsed:.2f}s total)"
        )
    else:
        print("\n  >>> All configurations timed out.")

    print("  PASSED")


def test_server_roundtrip(output_dir):
    """Test 4: Start server subprocess, connect, do reset + step."""
    print("\n=== Test 4: Server round-trip ===")
    port = 50099

    # Start server in subprocess
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "language_table.lamer.server_main",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--num_envs", "2",
            "--block_mode", "BLOCK_4",
            "--no_reward",
            "--max_inner_steps", "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    print("  Waiting for server to start...")
    time.sleep(10)

    try:
        # Minimal client using protocol
        from language_table.lamer.protocol import (
            EnvRequest, EnvResponse, send_message, recv_message,
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.connect(("127.0.0.1", port))
        print("  Connected to server")

        # get_properties
        send_message(sock, EnvRequest("1", "get_properties"))
        resp = recv_message(sock)
        print(f"  Properties: {resp.result}")
        assert resp.status == "ok"
        assert resp.result["num_processes"] == 2

        # reset
        send_message(sock, EnvRequest("2", "reset"))
        resp = recv_message(sock)
        assert resp.status == "ok"
        obs, infos = resp.result
        print(f"  Reset OK. Got {len(obs['text'])} text obs")

        # step
        send_message(
            sock,
            EnvRequest("3", "step", args=(["push red star to blue cube"] * 2,)),
        )
        resp = recv_message(sock)
        assert resp.status == "ok"
        print(f"  Step OK.")

        # close
        send_message(sock, EnvRequest("4", "close"))
        resp = recv_message(sock)
        sock.close()
        print("  PASSED")

    finally:
        server_proc.terminate()
        server_proc.wait(timeout=10)


def test_state_to_text():
    """Test 5: State-to-text conversion."""
    print("\n=== Test 5: State-to-text ===")
    from language_table.lamer.state_to_text import state_to_text

    # Construct a mock obs dict
    instruction = np.zeros(512, dtype=np.int32)
    text = "push the red star to the blue cube"
    encoded = list(text.encode("utf-8"))
    instruction[:len(encoded)] = encoded

    obs = {
        "instruction": instruction,
        "effector_translation": np.array([0.35, -0.12], dtype=np.float32),
        "block_red_star_translation": np.array([0.42, -0.15], dtype=np.float32),
        "block_red_star_mask": np.array([1.0], dtype=np.float32),
        "block_blue_cube_translation": np.array([0.30, 0.20], dtype=np.float32),
        "block_blue_cube_mask": np.array([1.0], dtype=np.float32),
        "block_green_moon_translation": np.array([0.55, 0.10], dtype=np.float32),
        "block_green_moon_mask": np.array([0.0], dtype=np.float32),  # not on table
    }

    result = state_to_text(obs)
    print(f"  Output:\n{result}")

    assert "Task: push the red star to the blue cube" in result
    assert "End-effector: (0.350, -0.120)" in result
    assert "red_star" in result
    assert "blue_cube" in result
    assert "green_moon" not in result  # masked out
    print("  PASSED")


def test_env_manager(output_dir):
    """Test 6: EnvironmentManager integration."""
    print("\n=== Test 6: EnvironmentManager ===")
    from language_table.lamer.envs import LanguageTableMultiProcessEnv
    from language_table.lamer.env_manager import LanguageTableEnvironmentManager

    envs = LanguageTableMultiProcessEnv(
        num_envs=2, block_mode="BLOCK_4", seed=42
    )
    manager = LanguageTableEnvironmentManager(
        envs=envs, max_inner_steps=5
    )

    obs, infos = manager.reset()
    print(f"  Reset OK. Text obs sample:\n    {obs['text'][0][:100]}...")

    goals = ["push the red star to the blue cube"] * 2
    obs, rewards, dones, infos = manager.step(goals, phase="play")
    print(f"  Step OK. Rewards: {rewards}, Dones: {dones}")

    # Restart
    obs, infos = manager.restart()
    print(f"  Restart OK.")

    manager.close()
    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="Language Table LaMer test suite")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("LT_RENDER_DIR", "/tmp/lt_renders"),
    )
    parser.add_argument("--scaling_sweep", action="store_true")
    parser.add_argument("--num_cpus", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                        help="num_cpus values to sweep in scaling test")
    parser.add_argument("--timeout", type=float, default=10,
                        help="Timeout (seconds) per ray.get call in scaling test")
    parser.add_argument("--test_server", action="store_true")
    parser.add_argument("--test_only", type=str, default=None,
                        help="Run only specific test (1-6)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tests = {
        "1": ("Single env smoke", lambda: test_single_env(args.output_dir)),
        "2": ("Parallel test", lambda: test_parallel(args.num_envs, args.num_steps, args.output_dir)),
        "3": ("Scaling sweep", lambda: test_scaling_sweep(args.output_dir, args.num_cpus, args.timeout)),
        "4": ("Server round-trip", lambda: test_server_roundtrip(args.output_dir)),
        "5": ("State-to-text", test_state_to_text),
        "6": ("Env manager", lambda: test_env_manager(args.output_dir)),
    }

    if args.test_only:
        to_run = [args.test_only]
    else:
        to_run = ["1", "5", "6", "2"]
        if args.scaling_sweep:
            to_run.append("3")
        if args.test_server:
            to_run.append("4")

    passed = 0
    failed = 0
    for test_id in to_run:
        name, fn = tests[test_id]
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
