"""MuJoCo <-> PX4 frame and geodetic conversion.

The only module in this repository allowed to rotate anything (plan section 8).

Conventions, all load-bearing (plan section 3.6):

* MuJoCo world is **ENU** (+x east, +y north, +z up). This is our model-authoring
  rule, not something MuJoCo enforces beyond z-up.
* MuJoCo body is **FLU** (x forward, y left, z up).
* PX4 world is **NED**, PX4 body is **FRD**.
* Both attitude quaternions are body->world, ``[w, x, y, z]``. The conversion is
  a change of basis, **not** a conjugation.

A MuJoCo identity attitude is therefore PX4 yaw **+90 deg**, and
``yaw_px4 = 90 deg - yaw_mujoco``. That is correct, not a bug.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ENU->NED: 180 deg about (1, 1, 0)/sqrt(2).
Q_NED_ENU: NDArray[np.float64] = np.array([0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0])
# FLU->FRD: 180 deg about x. Self-inverse as a rotation.
Q_FLU_FRD: NDArray[np.float64] = np.array([0.0, 1.0, 0.0, 0.0])

STANDARD_GRAVITY = 9.80665
# PX4's geo library uses a sphere, not WGS84 (src/lib/geo/geo.h:55).
EARTH_RADIUS_M = 6371000.0


def quat_mul(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product, ``[w, x, y, z]`` in and out."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: ArrayLike) -> NDArray[np.float64]:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([w, -x, -y, -z])


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return arr / norm


def quat_to_euler_321(q: ArrayLike) -> NDArray[np.float64]:
    """``[roll, pitch, yaw]`` in radians, PX4's 321 order (``matrix::Eulerf``)."""
    w, x, y, z = quat_normalize(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - x * z))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def euler_321_to_quat(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Inverse of :func:`quat_to_euler_321`."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def vec_enu_to_ned(v: ArrayLike) -> NDArray[np.float64]:
    """``(x_e, y_n, z_u) -> (n, e, d)``. Determinant +1, a proper rotation."""
    e, n, u = np.asarray(v, dtype=np.float64)
    return np.array([n, e, -u])


def vec_ned_to_enu(v: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`vec_enu_to_ned`; the map is its own inverse."""
    n, e, d = np.asarray(v, dtype=np.float64)
    return np.array([e, n, -d])


def vec_flu_to_frd(v: ArrayLike) -> NDArray[np.float64]:
    """Negate y and z. Applies to true vectors *and* to body rates: FLU->FRD is
    a proper rotation, so a pseudovector transforms identically (plan 3.6)."""
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([x, -y, -z])


# The map is an involution; the alias documents intent at call sites.
vec_frd_to_flu = vec_flu_to_frd


def mujoco_quat_to_px4(q_mujoco: ArrayLike) -> NDArray[np.float64]:
    """FLU->ENU body quaternion to PX4's FRD->NED body quaternion.

    ``q_px4 = q_ned_enu * q_mujoco * q_flu_frd^-1``. No conjugation of
    ``q_mujoco`` itself: both sides are body->world.
    """
    return quat_normalize(
        quat_mul(quat_mul(Q_NED_ENU, quat_normalize(q_mujoco)), quat_conj(Q_FLU_FRD))
    )


def px4_quat_to_mujoco(q_px4: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`mujoco_quat_to_px4` (up to overall sign)."""
    return quat_normalize(
        quat_mul(quat_mul(quat_conj(Q_NED_ENU), quat_normalize(q_px4)), Q_FLU_FRD)
    )


class GeodeticProjection:
    """Azimuthal equidistant projection, bit-compatible in intent with PX4's
    ``MapProjection`` (``src/lib/geo/geo.cpp``), including the spherical earth.

    PX4 latches its own reference from the first ``HIL_STATE_QUATERNION`` we
    send, so this class owns both sides of the local frame (plan section 3.6).
    """

    def __init__(self, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float = 0.0) -> None:
        self.ref_lat_deg = float(ref_lat_deg)
        self.ref_lon_deg = float(ref_lon_deg)
        self.ref_alt_m = float(ref_alt_m)
        self._ref_lat = math.radians(self.ref_lat_deg)
        self._ref_lon = math.radians(self.ref_lon_deg)
        self._sin_lat = math.sin(self._ref_lat)
        self._cos_lat = math.cos(self._ref_lat)

    def project(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Geodetic to local ``(north, east)`` metres."""
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        cos_d_lon = math.cos(lon - self._ref_lon)
        arg = max(-1.0, min(1.0, self._sin_lat * sin_lat + self._cos_lat * cos_lat * cos_d_lon))
        c = math.acos(arg)
        k = (c / math.sin(c)) if abs(c) > 0.0 else 1.0
        north = k * (self._cos_lat * sin_lat - self._sin_lat * cos_lat * cos_d_lon) * EARTH_RADIUS_M
        east = k * cos_lat * math.sin(lon - self._ref_lon) * EARTH_RADIUS_M
        return north, east

    def reproject(self, north_m: float, east_m: float) -> tuple[float, float]:
        """Local ``(north, east)`` metres to geodetic degrees."""
        x_rad = north_m / EARTH_RADIUS_M
        y_rad = east_m / EARTH_RADIUS_M
        c = math.sqrt(x_rad * x_rad + y_rad * y_rad)
        if abs(c) <= 0.0:
            return self.ref_lat_deg, self.ref_lon_deg
        sin_c, cos_c = math.sin(c), math.cos(c)
        lat = math.asin(cos_c * self._sin_lat + (x_rad * sin_c * self._cos_lat) / c)
        lon = self._ref_lon + math.atan2(
            y_rad * sin_c, c * self._cos_lat * cos_c - x_rad * self._sin_lat * sin_c
        )
        return math.degrees(lat), math.degrees(lon)

    def enu_to_geodetic(self, pos_enu: ArrayLike) -> tuple[float, float, float]:
        """MuJoCo ENU position (metres from origin) to ``(lat, lon, alt_m)``."""
        north, east, down = vec_enu_to_ned(pos_enu)
        lat, lon = self.reproject(north, east)
        return lat, lon, self.ref_alt_m - down


def rebase_ned(
    pos_ned: ArrayLike, source: GeodeticProjection, target: GeodeticProjection
) -> NDArray[np.float64]:
    """A NED position relative to ``source``'s datum, re-expressed relative to
    ``target``'s.

    For comparing two local frames that latched different origins: EKF2 picks its
    own from GNSS, ground truth uses the simulator's home. Raw x/y/z then differ by
    a near-constant offset that reads as estimator bias (``scripts/hover_error.py``).
    """
    north, east, down = (float(v) for v in np.asarray(pos_ned, dtype=np.float64))
    lat, lon = source.reproject(north, east)
    north, east = target.project(lat, lon)
    # alt = source.ref_alt - down_source = target.ref_alt - down_target
    return np.array([north, east, down + target.ref_alt_m - source.ref_alt_m])
