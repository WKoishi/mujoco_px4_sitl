"""HIL message encode/decode tests -- phase 1/2, no PX4 build required.

Round-trips go through pymavlink's own parser, so a framing or field-order
mistake shows up here rather than as an unexplained PX4 behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mavlink

from mujoco_px4_sitl import hil


def make_link() -> mavlink.MAVLink:
    link = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
    link.robust_parsing = True
    return link


def round_trip(payload: bytes) -> mavlink.MAVLink_message:
    messages = make_link().parse_buffer(payload)
    assert messages, "payload did not decode to a message"
    assert len(messages) == 1
    return messages[0]


# --- fields_updated bitmask ----------------------------------------------

def test_strategy_a_bitmask_value():
    """ACCEL | GYRO, from SimulatorMavlink.hpp:89-95."""
    assert hil.FIELD_ACCEL == 0x0007
    assert hil.FIELD_GYRO == 0x0038
    assert hil.FIELD_MAG == 0x01C0
    assert hil.FIELD_DIFF_PRESS == 0x0400
    assert hil.FIELD_BARO == 0x1A00
    assert hil.FIELDS_STRATEGY_A == 0x003F
    # Strategy A must not claim any sensor PX4's sensor_*_sim modules own.
    for owned_by_px4 in (hil.FIELD_MAG, hil.FIELD_BARO, hil.FIELD_DIFF_PRESS):
        assert hil.FIELDS_STRATEGY_A & owned_by_px4 == 0


# --- HIL_SENSOR ----------------------------------------------------------

def test_hil_sensor_round_trip():
    accel = np.array([0.12, -0.34, -9.81])
    gyro = np.array([0.01, -0.02, 0.03])
    msg = round_trip(hil.encode_hil_sensor(make_link(), 1_234_567, accel, gyro))
    assert msg.get_type() == "HIL_SENSOR"
    assert msg.time_usec == 1_234_567
    assert (msg.xacc, msg.yacc, msg.zacc) == pytest.approx(tuple(accel), abs=1e-6)
    assert (msg.xgyro, msg.ygyro, msg.zgyro) == pytest.approx(tuple(gyro), abs=1e-6)
    assert msg.fields_updated == hil.FIELDS_STRATEGY_A
    # id 0 drives PX4's clock and registers the lockstep component.
    assert msg.id == 0


def test_hil_sensor_leaves_unclaimed_fields_zero():
    """Unset groups are skipped entirely by PX4, so zeros are safe. temperature
    is only read under the BARO bit, hence 0 rather than an ambient value."""
    msg = round_trip(hil.encode_hil_sensor(make_link(), 1, np.zeros(3), np.zeros(3)))
    assert (msg.xmag, msg.ymag, msg.zmag) == (0.0, 0.0, 0.0)
    assert msg.abs_pressure == 0.0
    assert msg.diff_pressure == 0.0
    assert msg.pressure_alt == 0.0
    assert msg.temperature == 0.0


# --- HIL_STATE_QUATERNION units -----------------------------------------

def encode_state(**overrides):
    kwargs = dict(
        time_usec=42,
        q_px4=np.array([1.0, 0.0, 0.0, 0.0]),
        body_rates_frd=np.zeros(3),
        lat_deg=47.397742,
        lon_deg=8.545594,
        alt_m=488.0,
        vel_ned=np.zeros(3),
        accel_frd=np.zeros(3),
    )
    kwargs.update(overrides)
    return round_trip(hil.encode_hil_state_quaternion(make_link(), **kwargs))


def test_state_quaternion_geodetic_scaling():
    msg = encode_state()
    assert msg.lat == 473977420  # degE7
    assert msg.lon == 85455940
    assert msg.alt == 488000  # mm
    assert list(msg.attitude_quaternion) == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_state_quaternion_velocity_is_int16_cm_per_second():
    msg = encode_state(vel_ned=np.array([1.5, -2.25, 0.5]))
    assert (msg.vx, msg.vy, msg.vz) == (150, -225, 50)


def test_velocity_saturates_instead_of_wrapping():
    """A wrapped int16 velocity is a sign flip in ground truth, which reads as a
    fast climb turning into a descent."""
    msg = encode_state(vel_ned=np.array([500.0, -500.0, 400.0]))
    assert (msg.vx, msg.vy, msg.vz) == (32767, -32768, 32767)


def test_acceleration_is_milli_metres_per_second_squared():
    """common.xml documents mG, but PX4 divides by 1000 and uses the result as
    m/s^2 (SimulatorMavlink.cpp:618-621). Sending literal milli-g would be a
    9.81x error.
    """
    msg = encode_state(accel_frd=np.array([1.0, -2.0, -9.80665]))
    assert (msg.xacc, msg.yacc, msg.zacc) == (1000, -2000, -9807)
    # And PX4's own decode recovers the SI value.
    assert msg.zacc / 1000.0 == pytest.approx(-9.807, abs=1e-3)


def test_acceleration_saturates_at_about_3_3_g():
    """int16 at milli-m/s^2 is only +-32.767 m/s^2. Reachable in a contact
    impact, so saturate rather than wrap."""
    msg = encode_state(accel_frd=np.array([50.0, -50.0, 0.0]))
    assert (msg.xacc, msg.yacc) == (32767, -32768)


def test_body_rates_are_passed_through_in_radians():
    msg = encode_state(body_rates_frd=np.array([0.1, -0.2, 0.3]))
    assert (msg.rollspeed, msg.pitchspeed, msg.yawspeed) == pytest.approx(
        (0.1, -0.2, 0.3), abs=1e-6
    )


# --- HIL_ACTUATOR_CONTROLS ----------------------------------------------

def build_actuator_message(controls, mode, flags=hil.FLAG_LOCKSTEP):
    padded = list(controls) + [0.0] * (hil.NUM_ACTUATOR_OUTPUTS - len(controls))
    return mavlink.MAVLink_hil_actuator_controls_message(
        time_usec=99, controls=padded, mode=mode, flags=flags
    )


def test_decode_armed_controls():
    decoded = hil.decode_actuator_controls(build_actuator_message(
        [0.5, 0.5, 0.5, 0.5], hil.MODE_FLAG_ARMED | hil.MODE_FLAG_CUSTOM
    ))
    assert decoded.armed
    assert decoded.lockstep
    assert decoded.effective(4) == pytest.approx(np.full(4, 0.5))


def test_disarmed_forces_actuators_to_zero():
    """Treat "armed bit clear" as "all actuators zero" rather than trusting the
    payload (plan 3.5)."""
    decoded = hil.decode_actuator_controls(build_actuator_message(
        [0.9, 0.9, 0.9, 0.9], hil.MODE_FLAG_CUSTOM
    ))
    assert not decoded.armed
    assert decoded.effective(4) == pytest.approx(np.zeros(4))


def test_missing_lockstep_flag_is_visible():
    decoded = hil.decode_actuator_controls(build_actuator_message(
        [0.0] * 4, hil.MODE_FLAG_ARMED, flags=0
    ))
    assert not decoded.lockstep


def test_default_controls_are_a_safe_hold():
    """The loop applies last-known-good controls on a quiet frame, so the
    zero-initialised default must be inert."""
    assert hil.ActuatorControls().effective(4) == pytest.approx(np.zeros(4))
