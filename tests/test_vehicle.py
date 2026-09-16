"""Rotor model tests -- phase 4. MuJoCo only, no PX4 build required."""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_px4_sitl.config import Config
from mujoco_px4_sitl.frames import vec_flu_to_frd
from mujoco_px4_sitl.sim import MujocoPhysics
from mujoco_px4_sitl.vehicle import RotorModel, Vehicle

# Airframe geometry, mirroring px4/22001_mujoco_quad. Positions are FLU here.
ARM_XY = 0.17678
# Rotor index -> (forward, left). Order must match the airframe's CA_ROTOR*.
ROTOR_LAYOUT = {0: (+1, -1), 1: (-1, +1), 2: (+1, +1), 3: (-1, -1)}


@pytest.fixture
def physics() -> MujocoPhysics:
    return MujocoPhysics(Config())


@pytest.fixture
def vehicle(physics: MujocoPhysics) -> Vehicle:
    return physics.vehicle


def settle(vehicle: Vehicle, commands, dt: float = 10.0) -> None:
    """Drive the motor lag filter to steady state."""
    vehicle.update_commands(np.asarray(commands, dtype=np.float64), dt)


# --- geometry -------------------------------------------------------------

def test_rotor_sites_match_the_airframe_geometry(vehicle: Vehicle):
    """The MuJoCo sites and the airframe's CA_ROTOR* must describe the same
    points, up to the FLU/FRD y-sign. A mismatch shows up as slow yaw drift or
    roll/pitch coupling and is routinely misdiagnosed as an EKF fault.
    """
    assert vehicle.num_rotors == 4
    for index, (fwd, left) in ROTOR_LAYOUT.items():
        arm = vehicle.arm_body[index]
        assert arm[0] == pytest.approx(fwd * ARM_XY, abs=1e-6)
        assert arm[1] == pytest.approx(left * ARM_XY, abs=1e-6)


def test_spin_directions_alternate_across_the_diagonals(vehicle: Vehicle):
    """Diagonal pairs share a spin direction: (0, 1) CCW and (2, 3) CW."""
    spin = vehicle.params.spin
    assert (spin[0], spin[1]) == (+1, +1)
    assert (spin[2], spin[3]) == (-1, -1)


# --- hover ----------------------------------------------------------------

def test_hover_thrust_equals_weight(vehicle: Vehicle):
    settle(vehicle, np.full(4, vehicle.hover_command()))
    force, _ = vehicle.wrench_body()
    assert force[2] == pytest.approx(vehicle.weight, rel=1e-9)
    assert force[0] == pytest.approx(0.0) and force[1] == pytest.approx(0.0)


def test_hover_sits_near_mid_stick(vehicle: Vehicle):
    """Plan phase 4: total thrust at command 0.5 should be close to weight."""
    assert vehicle.hover_command() == pytest.approx(0.5, abs=0.02)
    settle(vehicle, np.full(4, 0.5))
    force, _ = vehicle.wrench_body()
    assert force[2] == pytest.approx(vehicle.weight, rel=0.05)


def test_full_command_gives_a_flyable_thrust_to_weight_ratio(vehicle: Vehicle):
    settle(vehicle, np.ones(4))
    force, _ = vehicle.wrench_body()
    assert 2.0 <= force[2] / vehicle.weight <= 5.0


def test_hover_torque_is_zero_on_every_axis(vehicle: Vehicle):
    """Equal commands on all four rotors: yaw torque sums to zero because the
    two CCW rotors cancel the two CW ones, and the lever arms cancel in pairs.
    """
    settle(vehicle, np.full(4, vehicle.hover_command()))
    _, torque = vehicle.wrench_body()
    assert torque == pytest.approx(np.zeros(3), abs=1e-9)


def test_yaw_torque_cancels_at_any_equal_command(vehicle: Vehicle):
    for command in (0.0, 0.25, 0.5, 0.75, 1.0):
        settle(vehicle, np.full(4, command))
        _, torque = vehicle.wrench_body()
        assert torque[2] == pytest.approx(0.0, abs=1e-9)


# --- control axes, in PX4's frame -----------------------------------------

def test_extra_thrust_on_the_right_rolls_right_side_up(vehicle: Vehicle):
    """Rotors 0 and 3 are the right-hand pair. More thrust there lifts the right
    side, which is negative roll torque in FRD."""
    settle(vehicle, [0.6, 0.5, 0.5, 0.6])
    _, torque_flu = vehicle.wrench_body()
    torque_frd = vec_flu_to_frd(torque_flu)
    assert torque_frd[0] < 0.0
    assert torque_frd[1] == pytest.approx(0.0, abs=1e-9)
    assert torque_frd[2] == pytest.approx(0.0, abs=1e-9)


def test_extra_thrust_at_the_front_pitches_nose_up(vehicle: Vehicle):
    """Rotors 0 and 2 are the front pair. Nose up is *positive* pitch in FRD,
    which the section 3.6 table confirms from the other direction: +30 degrees
    about MuJoCo body y (nose down) maps to PX4 pitch -30 degrees.
    """
    settle(vehicle, [0.6, 0.5, 0.6, 0.5])
    torque_frd = vec_flu_to_frd(vehicle.wrench_body()[1])
    assert torque_frd[1] > 0.0
    assert torque_frd[0] == pytest.approx(0.0, abs=1e-9)
    assert torque_frd[2] == pytest.approx(0.0, abs=1e-9)


def test_extra_thrust_on_the_ccw_pair_yaws_nose_right(vehicle: Vehicle):
    """CCW rotors (0, 1) drag the airframe clockwise seen from above, which is
    positive yaw in FRD. This is the sign PX4's ``CA_ROTOR*_KM > 0`` implies.
    """
    settle(vehicle, [0.6, 0.6, 0.5, 0.5])
    torque_frd = vec_flu_to_frd(vehicle.wrench_body()[1])
    assert torque_frd[2] > 0.0
    assert torque_frd[0] == pytest.approx(0.0, abs=1e-9)
    assert torque_frd[1] == pytest.approx(0.0, abs=1e-9)


def test_reaction_torque_follows_the_km_coefficient(vehicle: Vehicle):
    settle(vehicle, [1.0, 0.0, 0.0, 0.0])
    thrust = vehicle.state.thrust[0]
    torque = vehicle.wrench_body()[1]
    assert torque[2] == pytest.approx(-vehicle.params.km * thrust, rel=1e-9)


# --- actuator dynamics ----------------------------------------------------

def test_motor_lag_is_first_order_and_converges(vehicle: Vehicle):
    vehicle.reset()
    dt, tau = 0.001, vehicle.params.time_constant
    vehicle.update_commands(np.ones(4), dt)
    assert vehicle.state.command[0] == pytest.approx(dt / tau, rel=1e-9)
    for _ in range(int(10 * tau / dt)):
        vehicle.update_commands(np.ones(4), dt)
    assert vehicle.state.command == pytest.approx(np.ones(4), abs=1e-3)


def test_commands_are_clamped_to_the_motor_range(vehicle: Vehicle):
    """PWMSim normalizes motor functions to [0, 1]; anything outside is a bug
    upstream, not something to extrapolate into negative thrust."""
    settle(vehicle, [-0.5, 2.0, 0.5, 0.5])
    assert vehicle.state.command[0] == pytest.approx(0.0)
    assert vehicle.state.command[1] == pytest.approx(1.0)


def test_zero_command_produces_no_wrench(vehicle: Vehicle):
    settle(vehicle, np.zeros(4))
    force, torque = vehicle.wrench_body()
    assert force == pytest.approx(np.zeros(3))
    assert torque == pytest.approx(np.zeros(3))


# --- applied wrench bookkeeping -------------------------------------------

def test_mass_is_the_whole_subtree(vehicle: Vehicle):
    """Thrust must be calibrated against everything the rotors lift, not just
    the base body's own mass. Identical here (single body), which is why the
    distinction has to be asserted rather than observed: phase 7 hangs an arm
    off the base and the two diverge silently.
    """
    model = vehicle.model
    assert vehicle.total_mass == pytest.approx(
        float(model.body_subtreemass[vehicle.body_id])
    )


def test_apply_accumulates_and_clear_resets(physics: MujocoPhysics):
    """``apply`` adds to ``xfrc_applied`` so phase 7 can have other writers on
    the same body, which makes clearing the caller's job. Both halves matter: a
    missing clear sums the rotor wrench over every step of the frame.
    """
    vehicle, data = physics.vehicle, physics.data
    settle(vehicle, np.full(4, 0.5))
    body = vehicle.body_id

    vehicle.clear(data)
    vehicle.apply(data)
    once = np.array(data.xfrc_applied[body])
    vehicle.apply(data)
    twice = np.array(data.xfrc_applied[body])
    assert twice == pytest.approx(2.0 * once), "apply() overwrote instead of adding"
    assert np.any(once != 0.0)

    vehicle.clear(data)
    assert data.xfrc_applied[body] == pytest.approx(np.zeros(6))


def test_stepping_a_frame_does_not_accumulate_thrust(physics: MujocoPhysics):
    """The end-to-end guard on the above: a frame is many physics steps, so a
    missing clear would multiply the wrench by steps_per_frame."""
    vehicle = physics.vehicle
    command = np.full(4, vehicle.hover_command())
    settle(vehicle, command)
    physics.step_frame(command)
    applied = np.array(physics.data.xfrc_applied[vehicle.body_id, :3])
    # One frame's worth: the world-frame thrust magnitude is one vehicle weight
    # at hover command, not steps_per_frame times it.
    assert np.linalg.norm(applied) == pytest.approx(vehicle.weight, rel=1e-6)


# --- model preconditions --------------------------------------------------

# Minimal model with the right names but a hinge where the freejoint must be.
_HINGE_BASE_XML = """
<mujoco>
  <worldbody>
    <body name="base_link">
      <joint name="h" type="hinge" axis="0 0 1"/>
      <geom name="core" type="box" size="0.1 0.1 0.02" mass="1"/>
      <site name="imu"/>
      <site name="rotor0" pos="0.1 -0.1 0"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="imu_accel" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>
</mujoco>
"""

_NO_JOINT_XML = _HINGE_BASE_XML.replace(
    '<joint name="h" type="hinge" axis="0 0 1"/>', ""
)


def _physics_from_xml(tmp_path, xml: str) -> MujocoPhysics:
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return MujocoPhysics(Config(model_path=path))


def test_a_base_without_a_joint_is_rejected(tmp_path):
    """``body_jntadr`` is -1 for a jointless body, which would index the *last*
    joint rather than fail, and then read garbage as the vehicle pose."""
    with pytest.raises(ValueError, match="no joint"):
        _physics_from_xml(tmp_path, _NO_JOINT_XML)


def test_a_base_whose_joint_is_not_free_is_rejected(tmp_path):
    """The qpos/qvel slicing assumes the freejoint layout, so anything else is a
    silent misread rather than an error (plan 3.6)."""
    with pytest.raises(ValueError, match="not a freejoint"):
        _physics_from_xml(tmp_path, _HINGE_BASE_XML)


# --- closed-loop sanity in MuJoCo itself ----------------------------------

def test_hover_command_holds_altitude_in_mujoco(physics: MujocoPhysics):
    """Commanding hover thrust on all four rotors keeps the vehicle airborne
    (plan phase 4 exit criterion, the automatable half).
    """
    vehicle = physics.vehicle
    physics.data.qpos[physics.qpos_adr + 2] = 2.0
    physics.hold_pose()
    command = np.full(4, vehicle.hover_command())
    # Pre-charge the motor filter so the drop during spin-up is not counted.
    settle(vehicle, command)
    start = physics.data.qpos[physics.qpos_adr + 2]
    for _ in range(int(2.0 * physics.cfg.imu_rate_hz)):
        physics.step_frame(command)
    height = physics.data.qpos[physics.qpos_adr + 2]
    assert abs(height - start) < 0.10, f"drifted from {start:.3f} to {height:.3f} m"


def test_vehicle_stays_roughly_level_at_hover(physics: MujocoPhysics):
    vehicle = physics.vehicle
    physics.data.qpos[physics.qpos_adr + 2] = 2.0
    physics.hold_pose()
    command = np.full(4, vehicle.hover_command())
    settle(vehicle, command)
    for _ in range(int(2.0 * physics.cfg.imu_rate_hz)):
        physics.step_frame(command)
    rates = physics.state().gyro_frd
    assert np.max(np.abs(rates)) < 0.05, f"body rates grew to {rates}"
