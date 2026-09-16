"""Rotor thrust / torque model and actuator mapping.

Maps ``HIL_ACTUATOR_CONTROLS.controls[i]`` in ``[0, 1]`` to a force and a
reaction torque per rotor, applied at the rotor sites of the MuJoCo model.

Rotor indices are PX4's: ``controls[i]`` drives ``rotor{i}``, and the site
positions must match ``CA_ROTOR{i}_P*`` in the airframe file up to the FLU/FRD
y-sign (plan section 6, phase 4). A mismatch shows up as slow yaw drift or
roll/pitch cross-coupling, and is routinely misdiagnosed as an EKF fault.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mujoco
import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


@dataclass
class RotorModel:
    """Per-rotor parameters, shared by all rotors.

    Thrust is quadratic in the normalized command, which is the usual
    first-order approximation of ``T = C_T rho A R^2 omega^2`` with ``omega``
    proportional to command.

    ``k_thrust`` is the thrust at full command, per rotor, in newtons. Left at
    ``None`` it is calibrated from the model's own mass so that hover sits at
    ``hover_command`` (plan phase 4: hover near mid-stick). Note the quadratic
    plant and PX4's linear allocator agree in *slope* at ``u = 0.5``, so
    ``THR_MDL_FAC`` can stay at its default of 0.
    """

    k_thrust: float | None = None
    hover_command: float = 0.5
    # Torque = km * thrust, PX4's CA_ROTOR*_KM definition (module.yaml:211-218).
    km: float = 0.05
    # Spin direction per rotor index: +1 CCW about body +z (FLU), -1 CW.
    # Matches the KM signs in px4/22001_mujoco_quad.
    spin: tuple[int, ...] = (+1, +1, -1, -1)
    # First-order motor lag, seconds. 0 disables it.
    time_constant: float = 0.02
    site_prefix: str = "rotor"
    body_name: str = "base_link"


@dataclass
class RotorState:
    """Actuator state that persists across steps (the motor lag filter)."""

    command: NDArray[np.float64]
    thrust: NDArray[np.float64]


class Vehicle:
    """Applies actuator commands to a MuJoCo model as forces and torques."""

    def __init__(self, model: mujoco.MjModel, rotors: RotorModel | None = None) -> None:
        self.model = model
        self.params = rotors if rotors is not None else RotorModel()
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.params.body_name
        )
        if self.body_id < 0:
            raise ValueError(f"model has no body named {self.params.body_name!r}")

        self.site_ids: list[int] = []
        index = 0
        while True:
            sid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{self.params.site_prefix}{index}"
            )
            if sid < 0:
                break
            self.site_ids.append(sid)
            index += 1
        if not self.site_ids:
            raise ValueError(f"model has no {self.params.site_prefix}N sites")
        if len(self.params.spin) < self.num_rotors:
            raise ValueError(
                f"RotorModel.spin has {len(self.params.spin)} entries for "
                f"{self.num_rotors} rotors"
            )

        # Subtree, not this body alone: phase 7 hangs an arm off the base as
        # child bodies, and their mass is just as much what the rotors must
        # lift. Identical for a single-body model, so this changes nothing
        # today -- which is exactly why it belongs here rather than mid-phase-7,
        # where it would present as a hover thrust deficit and read as a rotor
        # model that needs tuning.
        self.total_mass = float(model.body_subtreemass[self.body_id])
        gravity = float(abs(model.opt.gravity[2]))
        self.weight = self.total_mass * gravity

        if self.params.k_thrust is None:
            # T(u) = k * u^2, and num_rotors * T(hover_command) = weight.
            u = self.params.hover_command
            self.k_thrust = self.weight / (self.num_rotors * u * u)
            _log.info(
                "calibrated k_thrust=%.3f N/rotor from mass=%.3f kg "
                "(hover at command %.2f, thrust/weight=%.2f at full)",
                self.k_thrust, self.total_mass, u,
                self.num_rotors * self.k_thrust / self.weight,
            )
        else:
            self.k_thrust = float(self.params.k_thrust)

        # Rotor positions relative to the body CoM, in the body frame. The lever
        # arm is about the CoM because that is where xfrc_applied acts.
        com_offset = np.asarray(model.body_ipos[self.body_id], dtype=np.float64)
        self.arm_body = np.array(
            [np.asarray(model.site_pos[sid], dtype=np.float64) - com_offset
             for sid in self.site_ids]
        )
        self.state = RotorState(
            command=np.zeros(self.num_rotors), thrust=np.zeros(self.num_rotors)
        )

    @property
    def num_rotors(self) -> int:
        return len(self.site_ids)

    def hover_command(self) -> float:
        """Normalized command at which total thrust equals weight."""
        return float(np.sqrt(self.weight / (self.num_rotors * self.k_thrust)))

    def reset(self) -> None:
        self.state.command[:] = 0.0
        self.state.thrust[:] = 0.0

    def update_commands(self, commands: NDArray[np.float64], dt: float) -> None:
        """Advance the motor lag filter by ``dt`` toward ``commands``."""
        target = np.clip(np.asarray(commands, dtype=np.float64)[: self.num_rotors], 0.0, 1.0)
        tau = self.params.time_constant
        if tau > 0.0 and dt > 0.0:
            alpha = min(1.0, dt / tau)
            self.state.command += alpha * (target - self.state.command)
        else:
            self.state.command[:] = target
        self.state.thrust = self.k_thrust * self.state.command ** 2

    def apply(self, data: mujoco.MjData) -> None:
        """Write the rotor wrench into ``data.xfrc_applied`` (world frame)."""
        thrust = self.state.thrust
        # Rotor thrust is along body +z (FLU) for every rotor.
        force_body = np.zeros(3)
        force_body[2] = float(np.sum(thrust))
        # Lever arms give roll/pitch; reaction torque gives yaw. A CCW rotor
        # (spin +1) drags the airframe about body -z.
        torque_body = np.cross(self.arm_body, np.column_stack(
            [np.zeros(self.num_rotors), np.zeros(self.num_rotors), thrust]
        )).sum(axis=0)
        spin = np.asarray(self.params.spin[: self.num_rotors], dtype=np.float64)
        torque_body[2] += float(-np.sum(spin * self.params.km * thrust))

        rot = np.asarray(data.xmat[self.body_id], dtype=np.float64).reshape(3, 3)
        # Accumulate rather than assign: xfrc_applied is a shared field, and
        # phase 7 will have other writers on this same body. The caller owns
        # clearing it once per step -- see clear().
        data.xfrc_applied[self.body_id, :3] += rot @ force_body
        data.xfrc_applied[self.body_id, 3:] += rot @ torque_body

    def clear(self, data: mujoco.MjData) -> None:
        """Zero this body's applied wrench. Call once before the writers."""
        data.xfrc_applied[self.body_id, :] = 0.0

    def wrench_body(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Current ``(force, torque)`` in the body frame. For tests."""
        thrust = self.state.thrust
        force = np.array([0.0, 0.0, float(np.sum(thrust))])
        torque = np.cross(self.arm_body, np.column_stack(
            [np.zeros(self.num_rotors), np.zeros(self.num_rotors), thrust]
        )).sum(axis=0)
        spin = np.asarray(self.params.spin[: self.num_rotors], dtype=np.float64)
        torque[2] += float(-np.sum(spin * self.params.km * thrust))
        return force, torque
