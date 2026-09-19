"""Rotor model tests -- phase 4. MuJoCo only, no PX4 build required."""

from __future__ import annotations

from pathlib import Path

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

_QUAD_XML = Path(Config().model_path).read_text()


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


def test_hover_sits_where_mpc_thr_hover_says_it_does(vehicle: Vehicle):
    """The idle offset moves hover off mid-stick, to 0.450 for the reference
    omega_idle / omega_max = 1/11 (MODELING_CONVENTIONS.md 2.5). That number is
    ``MPC_THR_HOVER`` in px4/22001_mujoco_quad; if this test moves, so must the
    airframe file, or hover is quietly mistuned.
    """
    assert vehicle.hover_command() == pytest.approx(0.450, abs=0.002)


def test_hover_command_is_mass_independent(tmp_path):
    """Auto-calibrated c_t holds thrust/weight at full command, so hover lands
    on the same command whatever the airframe weighs. That is what lets one
    ``MPC_THR_HOVER`` cover the quad, a heavier X8, and an arm on top of it.
    """
    heavy = _QUAD_XML.replace('mass="1.10"', 'mass="4.40"')
    physics = _physics_from_xml(tmp_path, heavy)
    assert physics.vehicle.total_mass > 4.0
    assert physics.vehicle.hover_command() == pytest.approx(0.450, abs=0.002)


def test_idle_thrust_is_small_but_not_zero(vehicle: Vehicle):
    """An armed motor at zero command still spins. 3.3 % of weight for the
    reference numbers -- enough to matter on the ground, not enough to fly.
    """
    settle(vehicle, np.zeros(4))
    force, _ = vehicle.wrench_body()
    assert force[2] / vehicle.weight == pytest.approx(0.033, abs=0.004)
    assert np.all(vehicle.state.omega == pytest.approx(vehicle.omega_idle))


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
    """Measured differentially against the all-idle state, because idle thrust
    on the other three rotors contributes yaw torque of its own now."""
    settle(vehicle, np.zeros(4))
    idle_torque = vehicle.wrench_body()[1][2]
    idle_thrust = vehicle.state.thrust[0]

    settle(vehicle, [1.0, 0.0, 0.0, 0.0])
    delta_thrust = vehicle.state.thrust[0] - idle_thrust
    delta_torque = vehicle.wrench_body()[1][2] - idle_torque
    assert delta_torque == pytest.approx(-vehicle.km[0] * delta_thrust, rel=1e-9)


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


def test_zero_command_produces_idle_thrust_and_no_torque(vehicle: Vehicle):
    """Idle thrust is equal on all rotors, so it cancels on every torque axis --
    the lever arms in pairs, the reaction torques by spin direction."""
    settle(vehicle, np.zeros(4))
    force, torque = vehicle.wrench_body()
    assert force[2] > 0.0
    assert force[0] == pytest.approx(0.0) and force[1] == pytest.approx(0.0)
    assert torque == pytest.approx(np.zeros(3), abs=1e-9)


def test_zero_omega_span_gives_a_genuinely_dead_rotor(physics: MujocoPhysics):
    """omega_idle = 0 is still allowed, and then zero command means zero thrust.
    The idle offset is a motor property, not something baked into the model.
    """
    vehicle = Vehicle(physics.model, RotorModel(omega_idle=0.0))
    settle(vehicle, np.zeros(4))
    force, torque = vehicle.wrench_body()
    assert force == pytest.approx(np.zeros(3))
    assert torque == pytest.approx(np.zeros(3))
    assert vehicle.hover_command() == pytest.approx(0.5, abs=1e-6)


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
      <site name="rotor1" pos="-0.1 0.1 0"/>
      <site name="rotor2" pos="0.1 0.1 0"/>
      <site name="rotor3" pos="-0.1 -0.1 0"/>
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


def _physics_from_xml(
    tmp_path, xml: str, rotors: RotorModel | None = None
) -> MujocoPhysics:
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return MujocoPhysics(Config(model_path=path), rotors)


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


# --- arbitrary rotor count: the coaxial X8 -------------------------------

# Eight rotor sites in MODELING_CONVENTIONS.md section 4's order, which is
# PX4's 12001_octo_cox converted to FLU. Coaxial pairs are (0,5), (1,4), (2,7),
# (3,6) -- not (0,4), (1,5). Upper deck z = 0.05, lower deck z = -0.05.
_X8_ROTORS = [
    (+0.35, -0.35, +0.05), (+0.35, +0.35, +0.05),
    (-0.35, +0.35, +0.05), (-0.35, -0.35, +0.05),
    (+0.35, +0.35, -0.05), (+0.35, -0.35, -0.05),
    (-0.35, -0.35, -0.05), (-0.35, +0.35, -0.05),
]
X8_SPIN = (+1, -1, +1, -1, +1, -1, +1, -1)
X8_PAIRS = [(0, 5), (1, 4), (2, 7), (3, 6)]

_X8_SITES = "\n".join(
    f'      <site name="rotor{i}" pos="{x} {y} {z}"/>'
    for i, (x, y, z) in enumerate(_X8_ROTORS)
)
_X8_XML = f"""
<mujoco>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="base_link">
      <freejoint/>
      <geom name="core" type="box" size="0.15 0.15 0.03" mass="3.0"/>
      <site name="imu" quat="1 0 0 0"/>
{_X8_SITES}
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="imu_accel" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def x8(tmp_path) -> Vehicle:
    return _physics_from_xml(tmp_path, _X8_XML, RotorModel(spin=X8_SPIN)).vehicle


def test_an_eight_rotor_model_loads(x8: Vehicle):
    """The whole point of step 1: this used to raise ValueError."""
    assert x8.num_rotors == 8
    assert x8.hover_command() == pytest.approx(0.450, abs=0.002)
    settle(x8, np.full(8, x8.hover_command()))
    force, torque = x8.wrench_body()
    assert force[2] == pytest.approx(x8.weight, rel=1e-6)
    assert torque == pytest.approx(np.zeros(3), abs=1e-9)


def test_default_quad_spin_is_rejected_on_an_eight_rotor_model(tmp_path):
    """A length mismatch is a config error, not something to truncate. Flying an
    X8 on four spin entries would present as yaw drift."""
    with pytest.raises(ValueError, match="spin has 4 entries for 8 rotors"):
        _physics_from_xml(tmp_path, _X8_XML)


def _command_for_thrust(vehicle: Vehicle, index: int, thrust: float) -> float:
    """Invert T = c_t * omega^2 and omega = idle + span * u, for one rotor."""
    omega = np.sqrt(thrust / vehicle.c_t[index])
    span = vehicle.omega_max[index] - vehicle.omega_idle
    return float((omega - vehicle.omega_idle) / span)


def test_a_coaxial_pair_produces_pure_yaw(x8: Vehicle):
    """Both rotors of a pair sit at the same (x, y), so a thrust difference
    between them cancels in roll and pitch and leaves only reaction torque. PZ
    contributes nothing: r x [0,0,T] ignores r's z component.

    The split is by *thrust*, not by command: thrust is quadratic in command, so
    equal command offsets do not cancel and would leak roll and pitch. That is
    the allocator's problem to solve, not the plant's -- PX4 mixes in the thrust
    domain for exactly this reason.
    """
    hover = x8.hover_command()
    settle(x8, np.full(8, hover))
    share = float(x8.state.thrust[0])

    for upper, lower in X8_PAIRS:
        assert x8.arm_body[upper][:2] == pytest.approx(x8.arm_body[lower][:2])
        assert x8.spin[upper] == -x8.spin[lower]

        command = np.full(8, hover)
        command[upper] = _command_for_thrust(x8, upper, 1.2 * share)
        command[lower] = _command_for_thrust(x8, lower, 0.8 * share)
        settle(x8, command)
        force, torque = x8.wrench_body()
        assert force[2] == pytest.approx(x8.weight, rel=1e-9)
        assert torque[0] == pytest.approx(0.0, abs=1e-9)
        assert torque[1] == pytest.approx(0.0, abs=1e-9)
        assert abs(torque[2]) > 1e-4


def test_equal_commands_cancel_reaction_torque_within_every_pair(x8: Vehicle):
    for command in (0.0, 0.25, 0.45, 1.0):
        settle(x8, np.full(8, command))
        thrust = x8.state.thrust
        for upper, lower in X8_PAIRS:
            pair_yaw = -(
                x8.spin[upper] * x8.km[upper] * thrust[upper]
                + x8.spin[lower] * x8.km[lower] * thrust[lower]
            )
            assert pair_yaw == pytest.approx(0.0, abs=1e-12)


def test_the_lower_deck_can_take_a_c_t_discount(x8: Vehicle):
    """Per-rotor c_t is the mechanism for coaxial interference (section 5). The
    discount value itself is still undecided, so only the mechanism is tested.
    """
    c_t = np.full(8, float(x8.c_t[0]))
    c_t[4:] *= 0.8
    discounted = Vehicle(x8.model, RotorModel(c_t=c_t, spin=X8_SPIN))
    settle(discounted, np.ones(8))
    upper = discounted.state.thrust[:4].sum()
    lower = discounted.state.thrust[4:].sum()
    assert lower == pytest.approx(0.8 * upper, rel=1e-9)
    # A weaker lower deck needs more command to hover, and the discount is
    # symmetric across the pairs, so it costs no attitude trim.
    assert discounted.hover_command() > x8.hover_command()
    settle(discounted, np.full(8, discounted.hover_command()))
    force, torque = discounted.wrench_body()
    assert force[2] == pytest.approx(discounted.weight, rel=1e-6)
    assert torque == pytest.approx(np.zeros(3), abs=1e-9)


def test_per_rotor_arrays_must_match_the_rotor_count(x8: Vehicle):
    with pytest.raises(ValueError, match="km has 3 entries for 8 rotors"):
        Vehicle(x8.model, RotorModel(km=[0.05, 0.05, 0.05], spin=X8_SPIN))


def test_scalars_broadcast_to_every_rotor(x8: Vehicle):
    vehicle = Vehicle(x8.model, RotorModel(km=0.07, omega_max=900.0, spin=X8_SPIN))
    assert vehicle.km == pytest.approx(np.full(8, 0.07))
    assert vehicle.omega_max == pytest.approx(np.full(8, 900.0))


def test_omega_is_clipped_to_the_motor_range(x8: Vehicle):
    vehicle = Vehicle(
        x8.model, RotorModel(omega_min=300.0, omega_idle=100.0, spin=X8_SPIN)
    )
    settle(vehicle, np.zeros(8))
    assert vehicle.state.omega == pytest.approx(np.full(8, 300.0))
    settle(vehicle, np.ones(8))
    assert vehicle.state.omega == pytest.approx(vehicle.omega_max)


def test_omega_max_below_omega_min_is_rejected(x8: Vehicle):
    with pytest.raises(ValueError, match="omega_max must exceed omega_min"):
        Vehicle(x8.model, RotorModel(omega_max=50.0, spin=X8_SPIN))


def test_omega_is_exposed_as_state_for_the_aero_layer(x8: Vehicle):
    """Every effect in section 5 reads omega, so it has to be inspectable rather
    than folded into the thrust the way k_thrust folded c_t and omega_max."""
    settle(x8, np.full(8, 0.5))
    omega = x8.state.omega
    expected = x8.omega_idle + (x8.omega_max - x8.omega_idle) * 0.5
    assert omega == pytest.approx(expected)
    assert x8.state.thrust == pytest.approx(x8.c_t * omega ** 2)


def test_build_physics_passes_the_rotor_model_through(tmp_path):
    """Previously build_physics never passed one, so an X8 could not be loaded
    through the normal entry point at all."""
    from mujoco_px4_sitl.sim import build_physics

    path = tmp_path / "x8.xml"
    path.write_text(_X8_XML)
    physics = build_physics(Config(model_path=path), RotorModel(spin=X8_SPIN))
    assert isinstance(physics, MujocoPhysics)
    assert physics.num_actuators == 8
    assert physics.vehicle.spin == pytest.approx(np.asarray(X8_SPIN))


# --- closed-loop sanity in MuJoCo itself, continued -----------------------

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
