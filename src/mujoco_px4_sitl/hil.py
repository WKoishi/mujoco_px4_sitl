"""HIL message encode / decode.

The only place where SI units become MAVLink's scaled integers (plan section 8).
Where ``common.xml``'s prose and PX4's actual decode disagree, PX4 wins and the
discrepancy is documented at the call site (plan section 3.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymavlink.dialects.v20 import common as mavlink

# HIL_SENSOR.fields_updated, from SimulatorMavlink.hpp:89-95. Per-sensor and
# independently checked, so unset groups may be left at zero.
FIELD_ACCEL = 0b0000000000111
FIELD_GYRO = 0b0000000111000
FIELD_MAG = 0b0000111000000
FIELD_DIFF_PRESS = 0b0010000000000
FIELD_BARO = 0b1101000000000
# Strategy A: we own accel+gyro, PX4's sensor_*_sim modules own the rest.
FIELDS_STRATEGY_A = FIELD_ACCEL | FIELD_GYRO  # 0x3F

# HIL_ACTUATOR_CONTROLS.mode (SimulatorMavlink.cpp:130-131).
MODE_FLAG_ARMED = 128
MODE_FLAG_CUSTOM = 1
# flags bit 0 is set iff PX4 was built with lockstep (SimulatorMavlink.cpp:134).
FLAG_LOCKSTEP = 1

NUM_ACTUATOR_OUTPUTS = 16
_INT16_MIN, _INT16_MAX = -32768, 32767


def _sat_int16(value: float) -> int:
    """Clamp then cast. A wrapped int16 is a sign flip in ground truth."""
    return int(max(_INT16_MIN, min(_INT16_MAX, round(float(value)))))


def encode_hil_sensor(
    link: mavlink.MAVLink,
    time_usec: int,
    accel_frd: ArrayLike,
    gyro_frd: ArrayLike,
    *,
    sensor_id: int = 0,
    fields_updated: int = FIELDS_STRATEGY_A,
) -> bytes:
    """Pack HIL_SENSOR. ``accel_frd`` in m/s^2 (specific force), ``gyro_frd`` in
    rad/s, both already in PX4's FRD body frame.

    ``sensor_id`` 0 is required: it is what drives PX4's clock and registers the
    lockstep component (plan section 3.2). It is a MAVLink v2 extension field, so
    a zero-trimmed message would decode to 0 anyway -- we set it explicitly
    rather than relying on that.
    """
    ax, ay, az = (float(v) for v in np.asarray(accel_frd, dtype=np.float64))
    gx, gy, gz = (float(v) for v in np.asarray(gyro_frd, dtype=np.float64))
    msg = mavlink.MAVLink_hil_sensor_message(
        time_usec=int(time_usec),
        xacc=ax, yacc=ay, zacc=az,
        xgyro=gx, ygyro=gy, zgyro=gz,
        # mag / baro / diff-pressure fields are ignored while their bits are
        # clear. temperature is only read under the BARO bit, so 0 is correct.
        xmag=0.0, ymag=0.0, zmag=0.0,
        abs_pressure=0.0, diff_pressure=0.0, pressure_alt=0.0, temperature=0.0,
        fields_updated=int(fields_updated),
        id=int(sensor_id),
    )
    return msg.pack(link)


def encode_hil_state_quaternion(
    link: mavlink.MAVLink,
    time_usec: int,
    q_px4: ArrayLike,
    body_rates_frd: ArrayLike,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    vel_ned: ArrayLike,
    accel_frd: ArrayLike,
) -> bytes:
    """Pack HIL_STATE_QUATERNION (ground truth for sensor_*_sim).

    Encoding traps, both verified against PX4's decode:

    * ``vx/vy/vz`` are int16 **cm/s** -> +-327.67 m/s.
    * ``xacc/yacc/zacc`` are documented as mG in ``common.xml``, but PX4 divides
      by 1000 and uses the result as m/s^2 (SimulatorMavlink.cpp:618-621). So we
      send **milli-m/s^2**; literal milli-g would be a 9.81x error. Being int16,
      full scale is only +-32.767 m/s^2 (~3.3 g), hence the saturation.
    """
    q = np.asarray(q_px4, dtype=np.float64)
    p, qq, r = (float(v) for v in np.asarray(body_rates_frd, dtype=np.float64))
    vn, ve, vd = (float(v) for v in np.asarray(vel_ned, dtype=np.float64))
    ax, ay, az = (float(v) for v in np.asarray(accel_frd, dtype=np.float64))
    msg = mavlink.MAVLink_hil_state_quaternion_message(
        time_usec=int(time_usec),
        attitude_quaternion=[float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        rollspeed=p, pitchspeed=qq, yawspeed=r,
        lat=int(round(lat_deg * 1e7)),
        lon=int(round(lon_deg * 1e7)),
        alt=int(round(alt_m * 1e3)),
        vx=_sat_int16(vn * 100.0),
        vy=_sat_int16(ve * 100.0),
        vz=_sat_int16(vd * 100.0),
        ind_airspeed=0, true_airspeed=0,
        xacc=_sat_int16(ax * 1000.0),
        yacc=_sat_int16(ay * 1000.0),
        zacc=_sat_int16(az * 1000.0),
    )
    return msg.pack(link)


@dataclass
class ActuatorControls:
    """Decoded HIL_ACTUATOR_CONTROLS.

    Motor functions are normalized ``[0, 1]``; servos and everything else
    ``[-1, 1]`` (PWMSim.cpp:60-98). PX4 zeroes the whole array while disarmed, so
    treat "armed bit clear" as "all actuators zero" rather than trusting the
    payload (plan section 3.5).
    """

    time_usec: int = 0
    controls: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NUM_ACTUATOR_OUTPUTS)
    )
    mode: int = 0
    flags: int = 0

    @property
    def armed(self) -> bool:
        return bool(self.mode & MODE_FLAG_ARMED)

    @property
    def lockstep(self) -> bool:
        return bool(self.flags & FLAG_LOCKSTEP)

    def effective(self, count: int) -> NDArray[np.float64]:
        """First ``count`` controls, forced to zero while disarmed."""
        if not self.armed:
            return np.zeros(count)
        out = np.zeros(count)
        n = min(count, len(self.controls))
        out[:n] = self.controls[:n]
        return out


def decode_actuator_controls(msg: mavlink.MAVLink_hil_actuator_controls_message) -> ActuatorControls:
    return ActuatorControls(
        time_usec=int(msg.time_usec),
        controls=np.asarray(msg.controls, dtype=np.float64),
        mode=int(msg.mode),
        flags=int(msg.flags),
    )
