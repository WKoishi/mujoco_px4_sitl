"""Frame conversion tests -- phase 2. No PX4 build, no simulator process.

The section 3.6 attitude table in IMPLEMENTATION_PLAN.md is the **specification**.
The expected values below are copied from it, not re-derived from the
implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mujoco_px4_sitl import frames


def axis_quat(axis: int, degrees: float) -> np.ndarray:
    """MuJoCo-frame quaternion: rotation of ``degrees`` about body axis 0/1/2."""
    half = math.radians(degrees) / 2.0
    vec = [0.0, 0.0, 0.0]
    vec[axis] = math.sin(half)
    return np.array([math.cos(half), *vec])


def px4_euler_deg(q_mujoco) -> np.ndarray:
    return np.degrees(frames.quat_to_euler_321(frames.mujoco_quat_to_px4(q_mujoco)))


# --- the section 3.6 attitude table, row by row ---------------------------

# (description, mujoco attitude, expected PX4 roll/pitch/yaw in degrees)
ATTITUDE_TABLE = [
    ("identity", np.array([1.0, 0.0, 0.0, 0.0]), (0.0, 0.0, 90.0)),
    ("+30 about body x", axis_quat(0, 30.0), (30.0, 0.0, 90.0)),
    ("-30 about body x", axis_quat(0, -30.0), (-30.0, 0.0, 90.0)),
    ("+30 about body y", axis_quat(1, 30.0), (0.0, -30.0, 90.0)),
    ("+30 about body z", axis_quat(2, 30.0), (0.0, 0.0, 60.0)),
]


@pytest.mark.parametrize(("name", "q_mujoco", "expected"), ATTITUDE_TABLE,
                         ids=[row[0] for row in ATTITUDE_TABLE])
def test_attitude_table(name, q_mujoco, expected):
    assert px4_euler_deg(q_mujoco) == pytest.approx(np.array(expected), abs=1e-9)


def test_roll_keeps_its_sign():
    """FLU->FRD is a rotation *about x*, so roll is invariant under it.

    A test asserting ``roll_px4 == -roll_mujoco`` passes only for a broken
    conversion, so assert the invariance explicitly.
    """
    for deg in (-60.0, -30.0, -5.0, 5.0, 30.0, 60.0):
        roll, pitch, yaw = px4_euler_deg(axis_quat(0, deg))
        assert roll == pytest.approx(deg, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw == pytest.approx(90.0, abs=1e-9)


def test_yaw_offset_and_sign():
    """``yaw_px4 = 90 deg - yaw_mujoco``."""
    for deg in (-90.0, -30.0, 0.0, 15.0, 45.0, 89.0):
        _, _, yaw = px4_euler_deg(axis_quat(2, deg))
        assert yaw == pytest.approx(90.0 - deg, abs=1e-9)


def test_pitch_flips_sign():
    for deg in (-40.0, -10.0, 10.0, 40.0):
        roll, pitch, yaw = px4_euler_deg(axis_quat(1, deg))
        assert pitch == pytest.approx(-deg, abs=1e-9)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert yaw == pytest.approx(90.0, abs=1e-9)


def test_body_x_rotation_discriminates_against_a_conjugated_conversion():
    """A conjugate inserted "to match conventions" swaps roll and pitch.

    Correct: ``roll +30, pitch 0, yaw +90``. Conjugated: ``roll 0, pitch -30,
    yaw -90``. Assert the full triple so any basis error is caught at once.
    """
    q = axis_quat(0, 30.0)
    assert px4_euler_deg(q) == pytest.approx(np.array([30.0, 0.0, 90.0]), abs=1e-9)

    # The classic error: conjugating the result, i.e. treating one side as
    # world->body. Roll and pitch swap places.
    conjugated = np.degrees(
        frames.quat_to_euler_321(frames.quat_conj(frames.mujoco_quat_to_px4(q)))
    )
    assert conjugated == pytest.approx(np.array([0.0, -30.0, -90.0]), abs=1e-9)

    # A different error with a different signature: conjugating the *input*
    # instead. Roll flips sign and pitch stays zero, which is what a test
    # asserting ``roll_px4 == -roll_mujoco`` would wrongly accept.
    conj_input = np.degrees(frames.quat_to_euler_321(frames.quat_mul(
        frames.quat_mul(frames.Q_NED_ENU, frames.quat_conj(q)),
        frames.quat_conj(frames.Q_FLU_FRD),
    )))
    assert conj_input == pytest.approx(np.array([-30.0, 0.0, 90.0]), abs=1e-9)


def test_outer_factors_are_involutions():
    """Both basis quaternions are 180 degree rotations, hence self-inverse.

    Worth pinning down: it means conjugating the *outer* factors of the sandwich
    is a no-op, so that particular slip is harmless and cannot be the cause when
    the attitude table disagrees. Only the input/result conjugations above bite.
    """
    for q_basis in (frames.Q_NED_ENU, frames.Q_FLU_FRD):
        identity = frames.quat_mul(q_basis, q_basis)
        assert np.allclose(np.abs(identity), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)


# --- round trips ----------------------------------------------------------

def test_quaternion_round_trip_up_to_sign():
    """``q`` and ``-q`` are the same rotation, so compare up to sign."""
    rng = np.random.default_rng(20240917)
    for _ in range(500):
        q = frames.quat_normalize(rng.normal(size=4))
        back = frames.px4_quat_to_mujoco(frames.mujoco_quat_to_px4(q))
        assert np.allclose(back, q, atol=1e-12) or np.allclose(back, -q, atol=1e-12)


def test_vector_conversions_are_involutions():
    rng = np.random.default_rng(7)
    for _ in range(100):
        v = rng.normal(size=3)
        assert np.allclose(frames.vec_ned_to_enu(frames.vec_enu_to_ned(v)), v)
        assert np.allclose(frames.vec_frd_to_flu(frames.vec_flu_to_frd(v)), v)


def test_frame_maps_are_proper_rotations():
    """Both have determinant +1, which is why they compose as quaternions and
    why angular velocity (a pseudovector) transforms like a true vector."""
    basis = np.eye(3)
    enu_ned = np.array([frames.vec_enu_to_ned(b) for b in basis]).T
    flu_frd = np.array([frames.vec_flu_to_frd(b) for b in basis]).T
    assert np.linalg.det(enu_ned) == pytest.approx(1.0)
    assert np.linalg.det(flu_frd) == pytest.approx(1.0)


def test_euler_quaternion_round_trip():
    rng = np.random.default_rng(11)
    for _ in range(200):
        roll = rng.uniform(-math.pi, math.pi)
        pitch = rng.uniform(-math.pi / 2 + 0.05, math.pi / 2 - 0.05)
        yaw = rng.uniform(-math.pi, math.pi)
        q = frames.euler_321_to_quat(roll, pitch, yaw)
        assert frames.quat_to_euler_321(q) == pytest.approx(
            np.array([roll, pitch, yaw]), abs=1e-9
        )


# --- accelerometer sign ---------------------------------------------------

def test_level_stationary_accel_is_minus_z_in_frd():
    """MuJoCo reads ``[0, 0, +9.81]`` in its z-up site frame; PX4 expects
    ``[0, 0, -9.81]``. Specific force at rest is the *negative* of gravity, so
    the FRD gravity vector is +z while the FRD accel reading is -z. Conflating
    the two is exactly the bug this test catches.
    """
    mujoco_reading = np.array([0.0, 0.0, frames.STANDARD_GRAVITY])
    frd = frames.vec_flu_to_frd(mujoco_reading)
    assert frd == pytest.approx(np.array([0.0, 0.0, -frames.STANDARD_GRAVITY]))


def test_free_fall_accel_is_zero():
    assert frames.vec_flu_to_frd(np.zeros(3)) == pytest.approx(np.zeros(3))


# --- angular velocity -----------------------------------------------------

def test_body_rates_negate_y_and_z():
    rates_flu = np.array([0.3, -0.7, 1.1])
    assert frames.vec_flu_to_frd(rates_flu) == pytest.approx(np.array([0.3, 0.7, -1.1]))


# --- geodetic projection --------------------------------------------------

# Reference origin: our own config default, which is also the origin PX4 adopts
# from our first HIL_STATE_QUATERNION -- so this test owns both sides.
REF_LAT, REF_LON, REF_ALT = 47.397742, 8.545594, 488.0


def test_projection_origin_maps_to_zero():
    proj = frames.GeodeticProjection(REF_LAT, REF_LON, REF_ALT)
    north, east = proj.project(REF_LAT, REF_LON)
    assert (north, east) == pytest.approx((0.0, 0.0), abs=1e-9)
    lat, lon = proj.reproject(0.0, 0.0)
    assert (lat, lon) == pytest.approx((REF_LAT, REF_LON), abs=1e-12)


def test_projection_round_trip():
    proj = frames.GeodeticProjection(REF_LAT, REF_LON, REF_ALT)
    for north, east in [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (-250.0, 375.0),
                        (5000.0, -5000.0)]:
        lat, lon = proj.reproject(north, east)
        assert proj.project(lat, lon) == pytest.approx((north, east), abs=1e-6)


def test_projection_matches_spherical_reference_points():
    """PX4's MapProjection is azimuthal equidistant on a sphere of R = 6371 km
    (``src/lib/geo/geo.h:55``), not WGS84. Check against hand-computed values on
    that sphere: 1 degree of latitude is exactly ``R * pi/180``.
    """
    proj = frames.GeodeticProjection(REF_LAT, REF_LON, REF_ALT)
    deg_m = frames.EARTH_RADIUS_M * math.pi / 180.0

    north, east = proj.project(REF_LAT + 1.0, REF_LON)
    assert north == pytest.approx(deg_m, rel=1e-9)
    assert east == pytest.approx(0.0, abs=1e-6)

    # Due east: the parallel shrinks by cos(latitude).
    north, east = proj.project(REF_LAT, REF_LON + 0.01)
    assert east == pytest.approx(
        deg_m * 0.01 * math.cos(math.radians(REF_LAT)), rel=1e-4
    )


def test_enu_to_geodetic_altitude_is_up_positive():
    proj = frames.GeodeticProjection(REF_LAT, REF_LON, REF_ALT)
    _, _, alt = proj.enu_to_geodetic([0.0, 0.0, 25.0])
    assert alt == pytest.approx(REF_ALT + 25.0)


def test_enu_to_geodetic_axes_are_not_transposed():
    """World +x is EAST and +y is NORTH (our model rule). A model authored
    x-north produces a clean 90 degree heading error that looks like a
    conversion bug but is not, so pin the axes down here.
    """
    proj = frames.GeodeticProjection(REF_LAT, REF_LON, REF_ALT)
    lat_e, lon_e, _ = proj.enu_to_geodetic([1000.0, 0.0, 0.0])
    lat_n, lon_n, _ = proj.enu_to_geodetic([0.0, 1000.0, 0.0])
    assert lon_e > REF_LON and lat_e == pytest.approx(REF_LAT, abs=2e-4)
    assert lat_n > REF_LAT and lon_n == pytest.approx(REF_LON, abs=1e-9)


def test_gyro_and_ground_truth_rates_share_one_conversion():
    """PX4 copies HIL_STATE_QUATERNION's body rates straight into
    ``vehicle_angular_velocity_groundtruth`` with no conversion of its own, so
    the FLU->FRD negation is entirely ours and must be identical on both paths.
    A ground truth that disagrees with the IMU reads as an estimator fault.
    """
    from mujoco_px4_sitl.config import Config
    from mujoco_px4_sitl.sim import MujocoPhysics

    physics = MujocoPhysics(Config())
    physics.data.qvel[physics.qvel_adr + 3:physics.qvel_adr + 6] = [0.2, -0.4, 0.6]
    import mujoco

    mujoco.mj_forward(physics.model, physics.data)
    state = physics.state()
    # sim.py feeds state.gyro_frd to both messages; assert the value itself is
    # the converted one, so the shared path cannot silently diverge.
    assert state.gyro_frd == pytest.approx(np.array([0.2, 0.4, -0.6]), abs=1e-9)
