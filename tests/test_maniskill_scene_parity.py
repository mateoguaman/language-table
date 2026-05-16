"""Hard-threshold checks for ManiSkill/PyBullet LanguageTable scene parity.

These tests intentionally focus on scene geometry, camera math, reset bounds,
and a small contact rollout. They do not validate rewards or the full PyBullet
observation API.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from language_table.environments import blocks, constants


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR = os.path.join(_REPO_ROOT, "language_table", "environments", "assets")
_BLOCKS_DIR = os.path.join(_ASSETS_DIR, "blocks")


def _load_obj_vertices(path: str) -> np.ndarray:
    vertices = []
    with open(path, "r", encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64)


def _quat_xyzw_to_wxyz(quat_xyzw):
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


class ManiSkillSceneParityTest(unittest.TestCase):
    def test_camera_pose_matches_pybullet_axes(self):
        from language_table.environments import maniskill_env

        pose = maniskill_env._compute_camera_pose()
        q = np.asarray(pose.q, dtype=np.float64)
        rot_mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        pb_rot = Rotation.from_euler("xyz", constants.CAMERA_ORIENTATION)
        expected_forward = pb_rot.apply([0.0, 0.0, 1.0])
        expected_up = pb_rot.apply([0.0, -1.0, 0.0])

        np.testing.assert_allclose(pose.p, constants.CAMERA_POSE, atol=1e-9)
        np.testing.assert_allclose(rot_mat[:, 0], expected_forward, atol=1e-6)
        np.testing.assert_allclose(rot_mat[:, 2], expected_up, atol=1e-6)

    def test_camera_intrinsics_preserve_pybullet_fov(self):
        from language_table.environments import maniskill_env

        policy_k = maniskill_env._camera_intrinsics(
            constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT
        )
        human_k = maniskill_env._camera_intrinsics(640, 360)

        self.assertLess(abs(policy_k[0, 0] - 0.803 * constants.IMAGE_WIDTH), 1e-5)
        self.assertLess(abs(policy_k[1, 1] - 0.803 * constants.IMAGE_WIDTH), 1e-5)
        self.assertAlmostEqual(policy_k[0, 2], constants.IMAGE_WIDTH / 2.0)
        self.assertAlmostEqual(policy_k[1, 2], constants.IMAGE_HEIGHT / 2.0)

        policy_fovy = 2.0 * math.atan((constants.IMAGE_HEIGHT / 2.0) / policy_k[1, 1])
        human_fovy = 2.0 * math.atan((360 / 2.0) / human_k[1, 1])
        self.assertLess(abs(policy_fovy - human_fovy), 1e-9)

    def test_workspace_and_block_mesh_extents_match_assets(self):
        plane_vertices = _load_obj_vertices(os.path.join(_ASSETS_DIR, "plane.obj"))
        plane_extents = plane_vertices.ptp(axis=0) * np.array([0.01524, 0.02032, 1.0])
        np.testing.assert_allclose(plane_extents[:2], [0.4572, 0.6096], atol=1e-6)

        cube_vertices = _load_obj_vertices(os.path.join(_BLOCKS_DIR, "cube.obj"))
        cube_extents = cube_vertices.ptp(axis=0)
        self.assertLess(abs(cube_extents[1] - 0.0381), 1e-5)
        self.assertGreater(cube_extents[0], 0.035)
        self.assertGreater(cube_extents[2], 0.035)

    def test_reset_samples_obey_workspace_thresholds(self):
        gym, maniskill_env = self._import_maniskill()
        env = gym.make(
            "LanguageTable-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pos",
            render_mode="rgb_array",
            num_envs=1,
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
            sim_backend="cpu",
            render_backend="cpu",
        )
        try:
            obs, _ = env.reset(seed=7)
            extra = obs["extra"]
            active_blocks = blocks.get_block_set(blocks.LanguageTableBlockVariants.BLOCK_4)
            xlo = constants.X_MIN + constants.WORKSPACE_BOUNDS_BUFFER
            xhi = constants.X_MAX - constants.WORKSPACE_BOUNDS_BUFFER
            ylo = constants.Y_MIN + constants.WORKSPACE_BOUNDS_BUFFER
            yhi = constants.Y_MAX - constants.WORKSPACE_BOUNDS_BUFFER

            xy_positions = []
            for block_name in active_blocks:
                xy = extra[f"block_{block_name}_translation"].cpu().numpy()[0]
                self.assertGreaterEqual(xy[0], xlo - 1e-4)
                self.assertLessEqual(xy[0], xhi + 1e-4)
                self.assertGreaterEqual(xy[1], ylo - 1e-4)
                self.assertLessEqual(xy[1], yhi + 1e-4)
                xy_positions.append(xy)

            for i, lhs in enumerate(xy_positions):
                for rhs in xy_positions[i + 1 :]:
                    self.assertGreater(
                        np.linalg.norm(lhs - rhs),
                        constants.BLOCK_DISTANCE_THRESHOLD - 1e-4,
                    )
        finally:
            env.close()

    def test_simple_push_rollout_tracks_pybullet_direction(self):
        ms_disp = self._run_maniskill_push()
        pb_disp = self._run_pybullet_push()

        self.assertGreater(pb_disp[0], 0.005)
        self.assertGreater(ms_disp[0], 0.005)
        self.assertLess(abs(ms_disp[0] - pb_disp[0]), 0.08)
        self.assertLess(abs(ms_disp[1] - pb_disp[1]), 0.05)

    def _import_maniskill(self):
        try:
            import gymnasium as gym
            import language_table.environments.maniskill_env as maniskill_env
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"ManiSkill scene unavailable: {exc}")
        return gym, maniskill_env

    def _run_maniskill_push(self):
        gym, _ = self._import_maniskill()
        import sapien
        import torch
        from mani_skill.utils.structs.pose import Pose

        env = gym.make(
            "LanguageTable-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pos",
            render_mode="rgb_array",
            num_envs=1,
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
            sim_backend="cpu",
            render_backend="cpu",
        )
        try:
            env.reset(seed=0)
            unwrapped = env.unwrapped
            device = unwrapped.device
            off_pose = Pose.create_from_pq(
                p=torch.tensor([[5.0, 5.0, 0.0]], device=device),
                q=torch.tensor(
                    [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                    device=device,
                ),
            )
            for actor in unwrapped._blocks.values():
                actor.set_pose(off_pose)

            block_name = "blue_cube"
            block_q = torch.tensor(
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                device=device,
            )
            start_xy = np.array([0.35, 0.0], dtype=np.float64)
            unwrapped._blocks[block_name].set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[start_xy[0], start_xy[1], 0.0]], device=device),
                    q=block_q,
                )
            )

            tool_q = _quat_xyzw_to_wxyz(
                Rotation.from_rotvec([0.0, math.pi, 0.0]).as_quat()
            )
            tool_z = 0.06
            for x in np.linspace(0.27, 0.43, 120):
                unwrapped.end_effector_tool.set_pose(
                    sapien.Pose(p=[float(x), 0.0, tool_z], q=tool_q)
                )
                unwrapped.scene.step()

            final_xy = (
                unwrapped._blocks[block_name].pose.p[..., :2].cpu().numpy()[0].copy()
            )
            return final_xy - start_xy
        finally:
            env.close()

    def _run_pybullet_push(self):
        import pybullet
        import pybullet_data

        client = pybullet.connect(pybullet.DIRECT)
        try:
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet.setGravity(0, 0, -9.8)
            pybullet.loadURDF("plane.urdf", basePosition=[0, 0, -0.001])
            pybullet.loadURDF(
                os.path.join(_ASSETS_DIR, "workspace_real.urdf"),
                basePosition=[0.35, 0.0, 0.0],
            )
            block_id = pybullet.loadURDF(os.path.join(_BLOCKS_DIR, "blue_cube.urdf"))
            start_xyz = [0.35, 0.0, 0.0]
            block_quat = pybullet.getQuaternionFromEuler([math.pi / 2, 0.0, 0.0])
            pybullet.resetBasePositionAndOrientation(block_id, start_xyz, block_quat)

            tool_collision = pybullet.createCollisionShape(
                pybullet.GEOM_CYLINDER,
                radius=0.0127,
                height=0.135,
                collisionFramePosition=[0.0, 0.0, 0.029],
            )
            tool_id = pybullet.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=tool_collision,
                basePosition=[0.27, 0.0, 0.06],
            )
            tool_quat = Rotation.from_rotvec([0.0, math.pi, 0.0]).as_quat()
            for x in np.linspace(0.27, 0.43, 120):
                pybullet.resetBasePositionAndOrientation(
                    tool_id, [float(x), 0.0, 0.06], tool_quat
                )
                pybullet.stepSimulation()

            final_xyz, _ = pybullet.getBasePositionAndOrientation(block_id)
            return np.asarray(final_xyz[:2], dtype=np.float64) - np.asarray(start_xyz[:2])
        finally:
            pybullet.disconnect(client)


if __name__ == "__main__":
    unittest.main()
