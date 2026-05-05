"""LanguageTable scene recreated in ManiSkill v3 (SAPIEN3).

Step 1: physical scene only — workspace, blocks, robot, camera, observations.
No reward logic is implemented here.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch
from scipy.spatial.transform import Rotation

from mani_skill.agents.robots.xarm6 import XArm6NoGripper
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    SceneConfig,
    SimConfig,
)

from language_table.environments import blocks as blocks_module
from language_table.environments import constants

# ---------------------------------------------------------------------------
# Asset paths — resolved relative to this file, not the cwd
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_THIS_DIR, "assets")
_BLOCKS_DIR = os.path.join(_ASSETS_DIR, "blocks")
_SUCTION_DIR = os.path.join(_ASSETS_DIR, "suction")

# PyBullet resets block bases at z=0 and lets the sim settle on the plane.
_BLOCK_Z = 0.0
_OFFSCREEN_BLOCK_XYZ = [5.0, 5.0, _BLOCK_Z]

_POLICY_CAMERA_WIDTH = constants.IMAGE_WIDTH
_POLICY_CAMERA_HEIGHT = constants.IMAGE_HEIGHT
_HUMAN_CAMERA_WIDTH = 640
_HUMAN_CAMERA_HEIGHT = 360

_PYBULLET_SIM_FREQ = 240
_PYBULLET_CONTROL_FREQ = 10
_PYBULLET_GRAVITY = [0.0, 0.0, -9.8]

_WORKSPACE_SCALE = [0.01524, 0.02032, 1.0]
_WORKSPACE_POSE = [0.35, 0.0, 0.0]
_WORKSPACE_COLOR = [0.2, 0.2, 0.2, 1.0]

_BLOCK_MASS = 0.01
_BLOCK_STATIC_FRICTION = 0.5
_BLOCK_DYNAMIC_FRICTION = 0.5
_BLOCK_RESTITUTION = 0.0

_TOOL_MASS = 0.1
_TOOL_RADIUS = 0.0127
_TOOL_HALF_LENGTH = 0.135 * 0.5
_TOOL_TIP_OFFSET = 0.029
_TOOL_COLOR = [0.5, 0.5, 0.5, 1.0]
_TOOL_HEAD_COLOR = [0.2, 0.2, 0.2, 1.0]


def _camera_intrinsics(width: int, height: int) -> np.ndarray:
    """Return the D415-like intrinsics used by the PyBullet renderer."""
    focal_len = 0.803 * width
    return np.array(
        [
            [focal_len, 0.0, width / 2.0],
            [0.0, focal_len, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _compute_camera_pose() -> sapien.Pose:
    """Convert PyBullet camera orientation (euler xyz) to a SAPIEN Pose.

    PyBullet convention: camera forward = local +Z, up = local -Y.
    SAPIEN convention:   camera forward = local +X, up = local +Z.
    """
    r = Rotation.from_euler("xyz", list(constants.CAMERA_ORIENTATION))
    pb_forward = r.apply([0.0, 0.0, 1.0])   # PyBullet +Z
    pb_up = r.apply([0.0, -1.0, 0.0])        # PyBullet -Y

    sapien_x = pb_forward
    sapien_z = pb_up
    sapien_y = np.cross(sapien_z, sapien_x)
    sapien_y /= np.linalg.norm(sapien_y)

    rot_mat = np.column_stack([sapien_x, sapien_y, sapien_z])
    q_xyzw = Rotation.from_matrix(rot_mat).as_quat()
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return sapien.Pose(p=list(constants.CAMERA_POSE), q=q_wxyz)


def _block_quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Compose rot_x(π/2) * rot_z(yaw) into a batch of wxyz quaternions.

    Original PyBullet block rotation is set as euler [pi/2, 0, yaw], which
    means: first rotate π/2 around X, then rotate yaw around Z.
    q_base = (cos π/4, sin π/4, 0, 0)  [wxyz, rot_x(pi/2)]
    q_yaw  = (cos y/2, 0, 0, sin y/2)  [wxyz, rot_z(yaw)]
    result = q_base * q_yaw
    """
    half_yaw = yaw * 0.5
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)

    # q_base components (constant)
    b = yaw.shape[0]
    w1 = yaw.new_full((b,), math.cos(math.pi / 4))
    x1 = yaw.new_full((b,), math.sin(math.pi / 4))
    y1 = yaw.new_zeros(b)
    z1 = yaw.new_zeros(b)

    # q_yaw components
    w2, x2, y2, z2 = cy, yaw.new_zeros(b), yaw.new_zeros(b), sy

    # Quaternion product
    qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([qw, qx, qy, qz], dim=-1)


@register_env("LanguageTable-v1", max_episode_steps=200)
class LanguageTableManiSkillEnv(BaseEnv):
    """LanguageTable tabletop environment ported to ManiSkill v3.

    Recreates the XArm6 robot, workspace, and colored blocks in SAPIEN.
    Observation includes overhead RGB camera, end-effector position, and
    per-block position/orientation.  Reward is not implemented here.

    Args:
        block_mode: which subset of blocks to activate each episode.
        robot_uids: must be "xarm6_nogripper".
    """

    SUPPORTED_ROBOTS = ["xarm6_nogripper"]
    agent: XArm6NoGripper

    def __init__(
        self,
        *args,
        robot_uids: str = "xarm6_nogripper",
        block_mode: blocks_module.LanguageTableBlockVariants = (
            blocks_module.LanguageTableBlockVariants.BLOCK_4
        ),
        **kwargs,
    ):
        self._block_mode = block_mode
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=_PYBULLET_SIM_FREQ,
            control_freq=_PYBULLET_CONTROL_FREQ,
            scene_config=SceneConfig(
                gravity=_PYBULLET_GRAVITY,
                contact_offset=0.005,
                rest_offset=0.0,
                solver_position_iterations=25,
                solver_velocity_iterations=4,
                enable_enhanced_determinism=True,
            ),
            default_materials_config=DefaultMaterialsConfig(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        build_ground(self.scene, altitude=-0.001)
        self._load_workspace()
        self._load_end_effector_tool()
        self._load_all_blocks()

    def _load_workspace(self):
        """Load the dark workspace surface.

        workspace_real.urdf has <?xml version="0.0"?> which lxml rejects, so
        we build the actor directly: visual from the plane.obj mesh (scaled to
        match the URDF). The URDF's collision geometry is commented out, so the
        PyBullet scene contacts the ground plane rather than a raised tabletop.
        plane.obj scale 0.01524 × 0.02032 → footprint 0.4572 m × 0.6096 m.
        """
        plane_obj = os.path.join(_ASSETS_DIR, "plane.obj")
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=plane_obj,
            scale=_WORKSPACE_SCALE,
            material=sapien.render.RenderMaterial(base_color=_WORKSPACE_COLOR),
        )
        builder.initial_pose = sapien.Pose(p=_WORKSPACE_POSE)
        self.workspace = builder.build_static(name="workspace")

    def _load_end_effector_tool(self):
        """Add the same cylinder_real-style tool visual/collider at the TCP."""
        tool_material = physx.PhysxMaterial(1.0, 1.0, 0.0)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=os.path.join(_SUCTION_DIR, "head.obj"),
            scale=[0.001, 0.001, 0.001],
            material=sapien.render.RenderMaterial(base_color=_TOOL_HEAD_COLOR),
        )
        builder.add_convex_collision_from_file(
            filename=os.path.join(_SUCTION_DIR, "head.obj"),
            scale=[0.001, 0.001, 0.001],
            material=tool_material,
        )
        tip_pose = sapien.Pose(p=[0.0, 0.0, _TOOL_TIP_OFFSET])
        builder.add_cylinder_visual(
            pose=tip_pose,
            radius=_TOOL_RADIUS,
            half_length=_TOOL_HALF_LENGTH,
            material=sapien.render.RenderMaterial(base_color=_TOOL_COLOR),
        )
        builder.add_cylinder_collision(
            pose=tip_pose,
            radius=_TOOL_RADIUS,
            half_length=_TOOL_HALF_LENGTH,
            material=tool_material,
        )
        builder.set_mass_and_inertia(_TOOL_MASS, sapien.Pose(), [1.0, 1.0, 1.0])
        builder.initial_pose = sapien.Pose()
        self.end_effector_tool = builder.build_kinematic(name="cylinder_real_tool")

    def _load_all_blocks(self):
        """Load all 16 block actors regardless of active set.

        Inactive blocks are kept off-screen at (5, 5) throughout the episode.
        This is required for GPU sim where actors cannot be added/removed after
        scene construction.
        """
        self._blocks: Dict[str, object] = {}
        block_material = physx.PhysxMaterial(
            _BLOCK_STATIC_FRICTION, _BLOCK_DYNAMIC_FRICTION, _BLOCK_RESTITUTION
        )

        for block_name in blocks_module.BLOCK_URDF_PATHS:
            urdf_path = os.path.join(_BLOCKS_DIR, f"{block_name}.urdf")
            loader = self.scene.create_urdf_loader()
            loader.name = block_name
            parsed = loader.parse(urdf_path, package_dir=_BLOCKS_DIR)
            builder = parsed["actor_builders"][0]
            for rec in builder.collision_records:
                rec.material = block_material
            builder.set_mass_and_inertia(_BLOCK_MASS, sapien.Pose(), [1.0, 1.0, 1.0])
            builder.initial_pose = sapien.Pose(
                p=_OFFSCREEN_BLOCK_XYZ,
                q=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
            )
            block_actor = builder.build(name=block_name)
            block_actor.set_linear_damping(0.0)
            block_actor.set_angular_damping(0.0001)
            self._blocks[block_name] = block_actor

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        K = _camera_intrinsics(_POLICY_CAMERA_WIDTH, _POLICY_CAMERA_HEIGHT)
        return [
            CameraConfig(
                uid="overhead_cam",
                pose=_compute_camera_pose(),
                width=_POLICY_CAMERA_WIDTH,
                height=_POLICY_CAMERA_HEIGHT,
                intrinsic=K,
                near=0.01,
                far=10.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        K = _camera_intrinsics(_HUMAN_CAMERA_WIDTH, _HUMAN_CAMERA_HEIGHT)
        return [
            CameraConfig(
                uid="render_camera",
                pose=_compute_camera_pose(),
                width=_HUMAN_CAMERA_WIDTH,
                height=_HUMAN_CAMERA_HEIGHT,
                intrinsic=K,
                near=0.01,
                far=10.0,
            )
        ]

    # ------------------------------------------------------------------
    # Action: accept 2D (Δx, Δy) and pad z=0 before forwarding to the
    # 3D pd_ee_delta_pos controller. The public action_space stays 3D
    # (as set by the controller); callers may pass 2D arrays here.
    # ------------------------------------------------------------------

    def step(self, action):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] == 2:
            action_3d = torch.zeros(
                (*action.shape[:-1], 3), device=self.device, dtype=torch.float32
            )
            action_3d[..., :2] = action
            action = action_3d
        return super().step(action)

    def _sync_end_effector_tool(self):
        if hasattr(self, "end_effector_tool"):
            self.end_effector_tool.set_pose(self.agent.tcp.pose)

    def _before_simulation_step(self):
        self._sync_end_effector_tool()

    def _after_control_step(self):
        self._sync_end_effector_tool()

    def _settle_scene(self, nsteps: int):
        for _ in range(nsteps):
            self._sync_end_effector_tool()
            self.scene.step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Reset robot to the LanguageTable home pose
            init_qpos = torch.tensor(
                constants.INITIAL_JOINT_POSITIONS, dtype=torch.float32
            ).unsqueeze(0).expand(b, -1)
            self.agent.reset(init_qpos)
            self._sync_end_effector_tool()

            # Active blocks for this episode
            active_blocks: List[str] = blocks_module.get_block_set(self._block_mode)

            # Move all blocks off-screen first
            off_q = torch.tensor(
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                dtype=torch.float32,
            )
            off_xyz = torch.tensor(_OFFSCREEN_BLOCK_XYZ, dtype=torch.float32)
            off_pose = Pose.create_from_pq(
                p=off_xyz.unsqueeze(0).expand(b, -1),
                q=off_q,
            )
            for block_actor in self._blocks.values():
                block_actor.set_pose(off_pose)
            self._settle_scene(nsteps=100)

            # Workspace sampling bounds (inset by WORKSPACE_BOUNDS_BUFFER)
            buf = constants.WORKSPACE_BOUNDS_BUFFER
            xlo = constants.X_MIN + buf
            xhi = constants.X_MAX - buf
            ylo = constants.Y_MIN + buf
            yhi = constants.Y_MAX - buf
            x_range = xhi - xlo
            y_range = yhi - ylo

            # Get current EE position to enforce ARM_DISTANCE_THRESHOLD
            ee_xy = self.agent.tcp.pose.p[:, :2]  # (b, 2)

            # placed_xy: list of already-placed positions, shape [(b,2), ...]
            placed_xys: List[torch.Tensor] = [ee_xy]

            for block_name in active_blocks:
                block_actor = self._blocks[block_name]

                # 20-attempt vectorised rejection sampling
                cand_xy = (
                    torch.rand((b, 2), device=self.device)
                    * torch.tensor([x_range, y_range], device=self.device)
                    + torch.tensor([xlo, ylo], device=self.device)
                )
                for _ in range(20):
                    new_cand = (
                        torch.rand((b, 2), device=self.device)
                        * torch.tensor([x_range, y_range], device=self.device)
                        + torch.tensor([xlo, ylo], device=self.device)
                    )
                    valid = torch.ones(b, dtype=torch.bool, device=self.device)
                    for prev in placed_xys:
                        dist = torch.linalg.norm(new_cand - prev, dim=-1)
                        threshold = (
                            constants.ARM_DISTANCE_THRESHOLD
                            if prev is placed_xys[0]
                            else constants.BLOCK_DISTANCE_THRESHOLD
                        )
                        valid &= dist > threshold
                    # Accept the new candidate for envs where it is valid
                    cand_xy = torch.where(valid.unsqueeze(-1), new_cand, cand_xy)

                yaw = torch.rand(b, device=self.device) * 2.0 * math.pi
                q = _block_quat_from_yaw(yaw)

                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = cand_xy
                xyz[:, 2] = _BLOCK_Z

                block_actor.set_pose(Pose.create_from_pq(p=xyz, q=q))
                placed_xys.append(cand_xy)

            self._settle_scene(nsteps=200)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: dict) -> dict:
        ee_pose = self.agent.tcp.pose
        obs: Dict[str, torch.Tensor] = {
            "effector_translation": ee_pose.p[..., :2],
        }

        active_blocks: List[str] = blocks_module.get_block_set(self._block_mode)
        for block_name in active_blocks:
            block_actor = self._blocks[block_name]
            block_pose = block_actor.pose
            obs[f"block_{block_name}_translation"] = block_pose.p[..., :2]

            # Yaw from wxyz quaternion: atan2(2(wz + xy), 1 - 2(y² + z²))
            q = block_pose.q  # (..., 4) wxyz
            w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            obs[f"block_{block_name}_orientation"] = yaw.unsqueeze(-1)

        return obs

    # ------------------------------------------------------------------
    # Required ManiSkill overrides (reward not implemented yet)
    # ------------------------------------------------------------------

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        }

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
