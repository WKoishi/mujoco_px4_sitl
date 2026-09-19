"""Rotor thrust / torque model and actuator mapping.

Maps ``HIL_ACTUATOR_CONTROLS.controls[i]`` in ``[0, 1]`` to a force and a
reaction torque per rotor, applied at the rotor sites of the MuJoCo model.

Rotor speed is an explicit state (MODELING_CONVENTIONS.md section 2.5)::

    omega_i = clip(omega_idle + (omega_max_i - omega_idle) * u_i,
                   omega_min_i, omega_max_i)
    T_i     = c_t_i * omega_i**2
    tau_z_i = -spin_i * km_i * T_i

Any rotor count is supported, and every coefficient is a per-rotor array, which
is what the coaxial X8's upper/lower asymmetry and the planned aerodynamic
effects both need -- all of them take omega, not a normalized command.

Rotor indices are PX4's: ``controls[i]`` drives ``rotor{i}``, and the site
positions must match ``CA_ROTOR{i}_P*`` in the airframe file up to the FLU/FRD
y-sign (plan section 6, phase 4). A mismatch shows up as slow yaw drift or
roll/pitch cross-coupling, and is routinely misdiagnosed as an EKF fault.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

# --- placeholder motor / propeller numbers --------------------------------
#
# No motor has been chosen for the X8 yet (MODELING_CONVENTIONS.md section 7
# lists what is still needed), so these stand in until the datasheet numbers
# land. They are not measurements. OMEGA_MAX_PLACEHOLDER is section 2.5's
# reference X8 figure; OMEGA_IDLE_PLACEHOLDER is one eleventh of it, which is
# what puts hover at command 0.450.
OMEGA_MAX_PLACEHOLDER = 1100.0
OMEGA_IDLE_PLACEHOLDER = 100.0
# Thrust coefficient a ~3 kg X8 needs to reach thrust/weight 4.0 at full
# command with the omega_max above. Only a yardstick for the auto-calibration
# warning below -- c_t itself defaults to None (auto-calibrated), because a
# fixed c_t makes hover mass-dependent and the flight baseline assumes 0.450.
CT_PLACEHOLDER = 3.0 * 9.80665 * 4.0 / (8.0 * OMEGA_MAX_PLACEHOLDER ** 2)

Scalar = float | Sequence[float] | NDArray[np.float64]


@dataclass
class RotorModel:
    """Rotor parameters. Every coefficient is scalar (all rotors) or per-rotor.

    A scalar broadcasts to the rotor count found in the model; a sequence must
    match it exactly. ``spin`` is the exception -- it is geometry, cannot be
    guessed for an arbitrary layout, and its length is what fixes the expected
    rotor count.

    ``c_t`` is the thrust coefficient in N/(rad/s)^2, i.e. ``T = c_t * omega^2``.
    Left at ``None`` it is calibrated from the model's own subtree mass so that
    full command gives ``thrust_to_weight`` (MODELING_CONVENTIONS.md section
    2.4's fallback, kept because a heavy arm should still fly with placeholder
    motors -- but it warns, since it fabricates motor capability).

    Note ``omega_idle > 0``: a real motor does not stop at zero command when
    armed. It moves hover off mid-stick, which is why ``MPC_THR_HOVER`` is 0.45
    and not 0.50 -- see px4/22001_mujoco_quad.
    """

    c_t: Scalar | None = None
    thrust_to_weight: float = 4.0
    omega_max: Scalar = OMEGA_MAX_PLACEHOLDER
    # Armed-but-idle speed, shared: it is an ESC setting, not a per-rotor one.
    omega_idle: float = OMEGA_IDLE_PLACEHOLDER
    # Lower clip on omega. None means omega_idle -- the armed floor.
    omega_min: Scalar | None = None
    # Torque = km * thrust, PX4's CA_ROTOR*_KM definition (module.yaml:211-218).
    km: Scalar = 0.05
    # Spin direction per rotor index: +1 CCW about body +z (FLU), -1 CW.
    # Matches the KM signs in px4/22001_mujoco_quad. Length = rotor count.
    spin: tuple[int, ...] = (+1, +1, -1, -1)
    # First-order motor lag, seconds. 0 disables it. Because omega is affine in
    # the command, filtering u and filtering omega are equivalent, so the filter
    # stays on the command (section 2.5).
    time_constant: float = 0.02
    site_prefix: str = "rotor"
    body_name: str = "base_link"


@dataclass
class RotorState:
    """Actuator state that persists across steps (the motor lag filter).

    ``omega`` is rad/s per rotor and is the quantity the aerodynamic layer reads
    (advance ratio, coaxial interference, gyroscopic precession).
    """

    command: NDArray[np.float64]
    omega: NDArray[np.float64]
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
        # Exact, not ">=": a spin tuple longer than the rotor count means the
        # model and the intended airframe disagree, and silently truncating it
        # is how an X8 config gets flown as a quad.
        if len(self.params.spin) != self.num_rotors:
            raise ValueError(
                f"RotorModel.spin has {len(self.params.spin)} entries for "
                f"{self.num_rotors} rotors found as "
                f"{self.params.site_prefix}0..{self.num_rotors - 1}"
            )

        self.spin = np.asarray(self.params.spin, dtype=np.float64)
        self.km = self._per_rotor(self.params.km, "km")
        self.omega_max = self._per_rotor(self.params.omega_max, "omega_max")
        self.omega_idle = float(self.params.omega_idle)
        self.omega_min = (
            np.full(self.num_rotors, self.omega_idle)
            if self.params.omega_min is None
            else self._per_rotor(self.params.omega_min, "omega_min")
        )
        if np.any(self.omega_max <= self.omega_min):
            raise ValueError("RotorModel.omega_max must exceed omega_min")

        # Subtree, not this body alone: phase 7 hangs an arm off the base as
        # child bodies, and their mass is just as much what the rotors must
        # lift. Identical for a single-body model, so this changes nothing
        # today -- which is exactly why it belongs here rather than mid-phase-7,
        # where it would present as a hover thrust deficit and read as a rotor
        # model that needs tuning.
        self.total_mass = float(model.body_subtreemass[self.body_id])
        gravity = float(abs(model.opt.gravity[2]))
        self.weight = self.total_mass * gravity

        if self.params.c_t is None:
            self.c_t = self._calibrate_c_t()
        else:
            self.c_t = self._per_rotor(self.params.c_t, "c_t")

        # Rotor positions relative to the body CoM, in the body frame. The lever
        # arm is about the CoM because that is where xfrc_applied acts.
        com_offset = np.asarray(model.body_ipos[self.body_id], dtype=np.float64)
        self.arm_body = np.array(
            [np.asarray(model.site_pos[sid], dtype=np.float64) - com_offset
             for sid in self.site_ids]
        )
        self.state = RotorState(
            command=np.zeros(self.num_rotors),
            omega=np.zeros(self.num_rotors),
            thrust=np.zeros(self.num_rotors),
        )
        self.reset()

    def _per_rotor(self, value: Scalar, name: str) -> NDArray[np.float64]:
        """Broadcast a scalar, or check a sequence's length."""
        array = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if array.size == 1:
            return np.full(self.num_rotors, float(array[0]))
        if array.size != self.num_rotors:
            raise ValueError(
                f"RotorModel.{name} has {array.size} entries for "
                f"{self.num_rotors} rotors"
            )
        return array.astype(np.float64, copy=True)

    def _calibrate_c_t(self) -> NDArray[np.float64]:
        """Fallback c_t from mass, split equally across rotors.

        MODELING_CONVENTIONS.md section 2.4: the upside is that any arm mass
        still flies; the downside is that the motors quietly get stronger, which
        hides a platform that would really be underpowered. Hence the warning.
        """
        share = self.weight * self.params.thrust_to_weight / self.num_rotors
        c_t = share / self.omega_max ** 2
        _log.warning(
            "c_t not given: calibrated to %.4g N/(rad/s)^2 per rotor from "
            "mass=%.3f kg for thrust/weight %.2f at full command (%.2f N per "
            "rotor at omega_max=%.0f rad/s). Placeholder motors, not measured "
            "ones -- a real platform may not deliver this.",
            float(c_t[0]), self.total_mass, self.params.thrust_to_weight,
            float(share), float(self.omega_max[0]),
        )
        ratio = float(np.max(c_t)) / CT_PLACEHOLDER
        if not 1 / 3 <= ratio <= 3:
            _log.warning(
                "calibrated c_t is %.1fx the reference X8 value (%.4g); the "
                "model's mass and omega_max are probably inconsistent",
                ratio, CT_PLACEHOLDER,
            )
        return c_t

    @property
    def num_rotors(self) -> int:
        return len(self.site_ids)

    def hover_command(self) -> float:
        """Normalized command at which total thrust equals weight.

        With ``omega`` affine in ``u``, total thrust is a quadratic in ``u``:
        ``sum c_t (idle + span u)^2 = weight``. Solved rather than approximated,
        because the idle offset moves the root off 0.5 (0.450 for the reference
        numbers) and that value is what ``MPC_THR_HOVER`` has to match.
        """
        span = self.omega_max - self.omega_idle
        a = float(np.sum(self.c_t * span ** 2))
        b = float(np.sum(2.0 * self.c_t * self.omega_idle * span))
        c = float(np.sum(self.c_t * self.omega_idle ** 2)) - self.weight
        if c >= 0.0:
            # Idle thrust alone already carries the weight -- not flyable, and
            # not something to paper over with a clamped command.
            _log.warning("idle thrust exceeds weight; hover command is 0")
            return 0.0
        u = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        if u > 1.0:
            _log.warning(
                "full command gives only thrust/weight %.2f; the vehicle cannot "
                "hover. Check the arm mass against c_t and omega_max",
                (a + b + c + self.weight) / self.weight,
            )
        return float(np.clip(u, 0.0, 1.0))

    def omega(self, command: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rotor speeds, rad/s, for a normalized command per rotor."""
        omega = self.omega_idle + (self.omega_max - self.omega_idle) * command
        return np.clip(omega, self.omega_min, self.omega_max)

    def reset(self) -> None:
        """Back to the armed-but-idle state: zero command, idle omega."""
        self.state.command[:] = 0.0
        self.state.omega = self.omega(self.state.command)
        self.state.thrust = self.c_t * self.state.omega ** 2

    def update_commands(self, commands: NDArray[np.float64], dt: float) -> None:
        """Advance the motor lag filter by ``dt`` toward ``commands``."""
        target = np.clip(np.asarray(commands, dtype=np.float64)[: self.num_rotors], 0.0, 1.0)
        tau = self.params.time_constant
        if tau > 0.0 and dt > 0.0:
            alpha = min(1.0, dt / tau)
            self.state.command += alpha * (target - self.state.command)
        else:
            self.state.command[:] = target
        self.state.omega = self.omega(self.state.command)
        self.state.thrust = self.c_t * self.state.omega ** 2

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
        torque_body[2] += float(-np.sum(self.spin * self.km * thrust))

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
        torque[2] += float(-np.sum(spin * self.km * thrust))
        return force, torque

    def describe_rotors(self) -> str:
        """One line per rotor: the numbers a model review has to check."""
        lines = [
            f"  {i}: c_t={self.c_t[i]:.4g} km={self.km[i]:.4g} "
            f"spin={int(self.spin[i]):+d} omega=[{self.omega_min[i]:.0f}, "
            f"{self.omega_max[i]:.0f}] rad/s  arm=("
            f"{self.arm_body[i][0]:+.4f}, {self.arm_body[i][1]:+.4f}, "
            f"{self.arm_body[i][2]:+.4f}) m"
            for i in range(self.num_rotors)
        ]
        return "\n".join(lines)
