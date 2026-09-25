"""MuJoCo model ownership, stepping, and sensor readout.

Everything leaving this module is already in **PX4 frames** -- the conversions
come from :mod:`frames`, and no rotation is written here (plan section 8).

Two implementations behind one interface: :class:`MujocoPhysics` and
:class:`StubPhysics`, the latter being the phase-1 hardcoded level-hover state
used to prove the protocol before trusting any dynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray

from . import frames
from .arm import ArmCommand, ArmServos, ArmStatus, JointReading, PropellerMonitor
from .config import Config
from .vehicle import RotorModel, Vehicle

_log = logging.getLogger(__name__)


@dataclass
class SimState:
    """One frame of vehicle state, in PX4 frames and SI units."""

    time: float = 0.0
    accel_frd: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    gyro_frd: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    q_px4: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    vel_ned: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    pos_ned: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0


class Physics(Protocol):
    """What :mod:`loop` requires of a physics backend."""

    num_actuators: int

    def step_frame(self, controls: NDArray[np.float64]) -> None: ...
    def state(self) -> SimState: ...
    def submit_arm_command(self, command: ArmCommand) -> None: ...
    def arm_status(self) -> ArmStatus | None: ...
    def arm_joints(self) -> JointReading: ...
    @property
    def time(self) -> float: ...


class MujocoPhysics:
    """Owns ``MjModel`` / ``MjData`` and advances one IMU frame at a time."""

    def __init__(self, cfg: Config, rotors: RotorModel | None = None) -> None:
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(Path(cfg.model_path)))
        # The physics rate is ours, not the model's: config owns the IMU/physics
        # ratio so it stays an integer (plan phase 3).
        self.model.opt.timestep = cfg.physics_dt
        self.data = mujoco.MjData(self.model)
        self.vehicle = Vehicle(self.model, rotors)
        self.steps_per_frame = cfg.steps_per_imu_frame
        self.projection = frames.GeodeticProjection(cfg.home_lat, cfg.home_lon, cfg.home_alt)

        self._body_id = self.vehicle.body_id
        self._accel_adr, self._accel_dim = self._sensor("imu_accel")
        self._gyro_adr, self._gyro_dim = self._sensor("imu_gyro")
        if self._accel_dim != 3 or self._gyro_dim != 3:
            raise ValueError("imu_accel / imu_gyro must be 3-axis sensors")
        # A body with no joint has body_jntadr == -1, which would index the last
        # joint instead of failing, and then read garbage as the vehicle pose.
        jnt_adr = int(self.model.body_jntadr[self._body_id])
        if jnt_adr < 0:
            raise ValueError(
                f"body {self.vehicle.params.body_name!r} has no joint; the base "
                f"must carry a freejoint (plan 3.6)"
            )
        jnt_type = int(self.model.jnt_type[jnt_adr])
        if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"body {self.vehicle.params.body_name!r} joint 0 is type "
                f"{jnt_type}, not a freejoint; qpos/qvel layout assumes "
                f"[x y z qw qx qy qz] / [vx vy vz wx wy wz] (plan 3.6)"
            )
        self.qpos_adr = int(self.model.jnt_qposadr[jnt_adr])
        self.qvel_adr = int(self.model.jnt_dofadr[jnt_adr])

        self._hold_pos = np.zeros(3)
        self._hold_quat: NDArray[np.float64] | None = None
        if cfg.inject_attitude is not None:
            roll, pitch, yaw = (np.radians(v) for v in cfg.inject_attitude)
            self._hold_quat = frames.euler_321_to_quat(roll, pitch, yaw)
        if cfg.hold_pose:
            self._hold_pos = np.array([0.0, 0.0, cfg.hold_height])
            self._pin_pose()
            _log.info(
                "holding pose at %.2f m, MuJoCo attitude %s deg (321 Euler); "
                "actuators ignored",
                cfg.hold_height, cfg.inject_attitude or (0.0, 0.0, 0.0),
            )

        mujoco.mj_forward(self.model, self.data)
        self.arm = ArmServos(self.model, cfg.arm_timeout_s, cfg.arm_on_timeout)
        self.arm.hold(self.data)
        self.propellers = PropellerMonitor(self.model, self._body_id, self.vehicle.site_ids)
        self.propellers.check(self.data)
        _log.info(
            "loaded %s: %d rotors, %.3f kg, physics %.0f Hz (%d steps per IMU frame)",
            cfg.model_path, self.vehicle.num_rotors, self.vehicle.total_mass,
            cfg.physics_rate_hz, self.steps_per_frame,
        )
        # Hover command is what MPC_THR_HOVER in the airframe file must match;
        # logging it makes a mismatch visible at boot instead of as altitude
        # oscillation in flight.
        _log.info(
            "rotors: hover command %.3f, thrust/weight %.2f at full command\n%s",
            self.vehicle.hover_command(),
            float(np.sum(self.vehicle.c_t * self.vehicle.omega_max ** 2))
            / self.vehicle.weight,
            self.vehicle.describe_rotors(),
        )
        if self.arm.count:
            _log.info(
                "arm: %d servo(s) %s0..%d driven by arm_cmd; watchdog %s",
                self.arm.count, "arm_act", self.arm.count - 1,
                f"{cfg.arm_timeout_s:g} s sim time, then {cfg.arm_on_timeout}"
                if cfg.arm_timeout_s > 0.0 else "off",
            )

    def _sensor(self, name: str) -> tuple[int, int]:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"model has no sensor named {name!r}")
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    @property
    def num_actuators(self) -> int:
        return self.vehicle.num_rotors

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step_frame(self, controls: NDArray[np.float64]) -> None:
        """Advance one IMU frame, holding ``controls`` for its whole duration."""
        dt = self.model.opt.timestep
        # data.ctrl is the arm's; the rotors go through xfrc_applied. Writing it
        # once per frame suffices, since MuJoCo holds ctrl across steps.
        self.arm.update(self.data)
        for _ in range(self.steps_per_frame):
            self.vehicle.update_commands(controls, dt)
            # Clear before the writers, because apply() accumulates. Without
            # this the rotor wrench would sum over every step of the frame.
            self.vehicle.clear(self.data)
            self.vehicle.apply(self.data)
            mujoco.mj_step(self.model, self.data)
        if self.cfg.hold_pose:
            # Phase 3: actuators still ignored, so ground truth moves only in
            # ways we dictate. Re-pin after the steps, before the sensor read.
            self._pin_pose()
        # Sensor values are stage-acc/vel quantities; refresh them after the
        # last step so the IMU reading belongs to the frame we just finished.
        mujoco.mj_forward(self.model, self.data)
        # On the pose the arm reached, which mj_forward just made current.
        self.propellers.check(self.data)

    def submit_arm_command(self, command: ArmCommand) -> None:
        """Hand an ``arm_cmd`` to the servos; it drives from the next ``step_frame``."""
        self.arm.submit(command, self.time)

    def arm_status(self) -> ArmStatus | None:
        """None for a model with neither arm servos nor a clearance check."""
        if not self.arm.count and not self.propellers.enabled:
            return None
        return ArmStatus(
            cmd_seq=None if self.arm.command is None else self.arm.command.seq,
            cmd_age=self.arm.age(self.time),
            cmd_stale=self.arm.stale,
            prop_clearance=self.propellers.clearance if self.propellers.enabled else None,
            accepted=self.arm.accepted,
            rejected=self.arm.rejected,
            clamped=self.arm.clamped,
            stale_episodes=self.arm.stale_episodes,
            intrusions=self.propellers.intrusions,
            deepest=self.propellers.deepest,
        )

    def arm_joints(self) -> JointReading:
        """The arm's joints now; empty arrays for a model with no arm servos."""
        return self.arm.joints(self.data)

    def state(self) -> SimState:
        qpos = self.data.qpos[self.qpos_adr:self.qpos_adr + 7]
        qvel = self.data.qvel[self.qvel_adr:self.qvel_adr + 6]
        accel_flu = np.asarray(self.data.sensordata[self._accel_adr:self._accel_adr + 3])
        gyro_flu = np.asarray(self.data.sensordata[self._gyro_adr:self._gyro_adr + 3])

        accel_frd = frames.vec_flu_to_frd(accel_flu)
        # qvel[3:6] is body-frame angular velocity (same as the gyro sensor), so
        # the IMU and ground-truth body rates go through one identical path.
        gyro_frd = frames.vec_flu_to_frd(gyro_flu)
        q_px4 = frames.mujoco_quat_to_px4(qpos[3:7])
        # qvel[0:3] is world-frame (ENU) linear velocity.
        vel_ned = frames.vec_enu_to_ned(qvel[0:3])
        pos_ned = frames.vec_enu_to_ned(qpos[0:3])
        lat, lon, alt = self.projection.enu_to_geodetic(qpos[0:3])
        return SimState(
            time=self.time, accel_frd=accel_frd, gyro_frd=gyro_frd, q_px4=q_px4,
            vel_ned=vel_ned, pos_ned=pos_ned, lat_deg=lat, lon_deg=lon, alt_m=alt,
        )

    def hold_pose(self, height: float | None = None) -> None:
        """Pin the vehicle in place (phase 3: open loop, known ground truth)."""
        if height is not None:
            self.data.qpos[self.qpos_adr + 2] = height
        self.data.qvel[self.qvel_adr:self.qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _pin_pose(self) -> None:
        """Restore the held pose and zero the velocities, then refresh sensors.

        The accelerometer then reads level-and-stationary specific force, which
        is what makes the injected attitude the only thing under test.
        """
        self.data.qpos[self.qpos_adr:self.qpos_adr + 3] = self._hold_pos
        if self._hold_quat is not None:
            self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = self._hold_quat
        self.data.qvel[self.qvel_adr:self.qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # An unsupported airborne body has qacc = g, so the accelerometer would
        # read free fall -- wrong for something we are claiming is stationary.
        # Zero the acceleration and recompute the acceleration-stage sensors, so
        # the IMU reads R^T * [0, 0, g]: what a real IMU on a stationary tilted
        # vehicle in a test rig reads. The gyro is a velocity-stage sensor and
        # already reads zero from qvel.
        self.data.qacc[self.qvel_adr:self.qvel_adr + 6] = 0.0
        mujoco.mj_rnePostConstraint(self.model, self.data)
        mujoco.mj_sensorAcc(self.model, self.data)

    def set_attitude(self, q_mujoco: NDArray[np.float64]) -> None:
        """Inject a MuJoCo-frame attitude (phase 3 attitude-table check)."""
        self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = frames.quat_normalize(q_mujoco)
        mujoco.mj_forward(self.model, self.data)


class StubPhysics:
    """Phase 1: level, stationary, no ``mj_step``.

    The constants are stated **directly in PX4 frames** -- identity attitude here
    means heading north and level in FRD->NED, not the conversion of a MuJoCo
    identity (which would be yaw +90 deg). Phase 1 deliberately does not reason
    through MuJoCo at all (plan phase 1).
    """

    num_actuators = 4

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._time = 0.0
        self._dt = cfg.imu_dt
        self._warned_arm = False

    @property
    def time(self) -> float:
        return self._time

    def step_frame(self, controls: NDArray[np.float64]) -> None:  # noqa: ARG002
        self._time += self._dt

    def submit_arm_command(self, command: ArmCommand) -> None:  # noqa: ARG002
        if not self._warned_arm:
            self._warned_arm = True
            _log.warning("stub physics has no arm; arm_cmd is ignored")

    def arm_status(self) -> ArmStatus | None:
        return None

    def arm_joints(self) -> JointReading:
        return JointReading(q=np.zeros(0), qd=np.zeros(0))

    def state(self) -> SimState:
        return SimState(
            time=self._time,
            # Level and stationary in FRD: specific force points "up", i.e. -z.
            accel_frd=np.array([0.0, 0.0, -frames.STANDARD_GRAVITY]),
            gyro_frd=np.zeros(3),
            q_px4=np.array([1.0, 0.0, 0.0, 0.0]),
            vel_ned=np.zeros(3),
            pos_ned=np.zeros(3),
            lat_deg=self.cfg.home_lat,
            lon_deg=self.cfg.home_lon,
            alt_m=self.cfg.home_alt,
        )


def build_physics(cfg: Config, rotors: RotorModel | None = None) -> Physics:
    """Build the physics backend. ``rotors`` is ignored by the stub.

    Per-rotor arrays arrive through here: an 8-rotor model needs an 8-entry
    ``spin``, and the default quad tuple is rejected against it rather than
    truncated. ``main.py`` builds the ``RotorModel`` from the sidecar named by
    ``--rotors`` / ``MUJOCO_SITL_ROTORS`` (:mod:`rotorconfig`), or an in-process
    caller passes one it built itself.
    """
    if cfg.stub_physics:
        return StubPhysics(cfg)
    return MujocoPhysics(cfg, rotors)
