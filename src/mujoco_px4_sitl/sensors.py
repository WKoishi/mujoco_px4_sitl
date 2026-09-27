"""Simulator-side sensor errors: what the vehicle's sensors add to the truth.

Under MAVLink HIL the simulator owns IMU errors: PX4 adds none on the
``HIL_SENSOR`` path (plan section 3.4), so an ideal reading reaches EKF2 as a
perfect IMU. :class:`ImuModel` adds one part's error budget to the accel and
gyro readings of that message **only**. Ground truth -- ``HIL_STATE_QUATERNION``,
the side channel, the recorder's state -- stays clean.

Per axis, in FRD, a reading is the truth plus

* a **turn-on bias**, drawn once per run;
* a **drift**, a first-order Gauss-Markov process starting from zero;
* **white noise**, one draw per IMU frame, of ``density * sqrt(rate / 2)``:
  the datasheet's RMS rows are density times ``sqrt(bandwidth)``, and a mean
  over one frame has a noise bandwidth of half the frame rate. Gazebo's
  ``density / sqrt(dt)`` is ``sqrt(2)`` more.

then **quantised** as a 16-bit output at PX4's simulated FIFO scale, 0.061 deg/s
and 0.49 mg per LSB, saturating at full scale. PX4's decode truncates toward
zero (``SimulatorMavlink.cpp:222-224, 265-267``), which is a dead band of
+-1 LSB and half an LSB of bias: the gyro's white noise per frame is half an
LSB, so at rest nearly all of it would vanish. So the value sent is the rounded
count plus half an LSB away from zero, which that truncation turns back into the
rounded count.

Not modelled: scale factor, cross-axis sensitivity, misalignment, temperature
(PX4's IMU temperature stays constant under strategy A), and **vibration**,
which on a flying multirotor is most of what an IMU reads besides the motion: a
datasheet's noise is the floor.

**Seeded, and drawn per frame**: a fixed number of draws per reading, so a seed
gives the same errors frame for frame whatever the wall clock did. The turn-on
biases, the accel and the gyro draw from separate streams of one seed, so
changing one's parameters does not shift the others. Strategy B's baro, mag and
GPS (plan section 3.3) would join this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frames import STANDARD_GRAVITY

_DEG = math.pi / 180.0
_MG = STANDARD_GRAVITY * 1e-3

# PX4's id-0 HIL_SENSOR scale, the simulated FIFO's (SimulatorMavlink.cpp:207, 250):
# a 16-bit part at +-16 g and +-2000 deg/s.
ACCEL_LSB = STANDARD_GRAVITY / 2048.0
GYRO_LSB = math.radians(2000.0 / 32768.0)
_FULL_SCALE_COUNT = 32767

Axes = float | tuple[float, float, float]


def quantise(value: ArrayLike, lsb: float) -> NDArray[np.float64]:
    """Round to ``lsb`` and saturate at full scale, then add half an LSB away
    from zero, so PX4's truncating decode yields the rounded count."""
    count = np.clip(np.round(np.asarray(value, dtype=np.float64) / lsb),
                    -_FULL_SCALE_COUNT, _FULL_SCALE_COUNT)
    return (count + 0.5 * np.sign(count)) * lsb


@dataclass(frozen=True)
class ImuSpec:
    """One IMU's error budget. Each value is a scalar or one per FRD axis;
    biases and drift are 1 sigma."""

    name: str
    gyro_noise: Axes  # rad/s/sqrt(Hz), one-sided
    gyro_turn_on: Axes  # rad/s
    gyro_drift: Axes  # rad/s, steady state
    accel_noise: Axes  # m/s^2/sqrt(Hz), one-sided
    accel_turn_on: Axes  # m/s^2
    accel_drift: Axes  # m/s^2, steady state
    drift_tau: float  # s, the drift's correlation time


# TDK InvenSense DS-000347 v1.6, Tables 1 and 2, typical, board level: the IMU of
# current Pixhawk-class boards, assumed until the vehicle's flight controller is
# chosen. Noise is the datasheet's. The biases are what PX4 flies with:
# * gyro turn-on, the initial ZRO tolerance, uncalibrated: as on the vehicle,
#   PX4's gyro_calibration removes it before arming when its norm exceeds
#   0.01 rad/s, and EKF2 learns what is left in flight;
# * accel turn-on, not the 20 mg zero-g tolerance: PX4 flies a calibrated
#   accelerometer, and what calibration leaves is the zero-g level's change since
#   then, 0.15 mg/degC over a chosen 20 degC;
# * drift, chosen: the datasheet gives no in-run stability. Its temperature
#   coefficients (ZRO 0.005 deg/s/degC, zero-g 0.15 mg/degC) over 10 degC, with
#   a thermal time constant of 300 s.
ICM_42688_P = ImuSpec(
    name="ICM-42688-P",
    gyro_noise=0.0028 * _DEG,
    gyro_turn_on=0.5 * _DEG,
    gyro_drift=0.05 * _DEG,
    accel_noise=(65e-6 * STANDARD_GRAVITY, 65e-6 * STANDARD_GRAVITY,
                 70e-6 * STANDARD_GRAVITY),
    accel_turn_on=3.0 * _MG,
    accel_drift=1.5 * _MG,
    drift_tau=300.0,
)

# What --imu accepts. "ideal" sends the truth, as every baseline before
# 2026-09-27 was flown.
IMU_MODELS: dict[str, ImuSpec | None] = {"icm42688p": ICM_42688_P, "ideal": None}


def _axes(value: ArrayLike) -> NDArray[np.float64]:
    return np.array(np.broadcast_to(np.asarray(value, dtype=np.float64), 3))


class _Triad:
    """One three-axis sensor's errors."""

    def __init__(self, noise: Axes, turn_on: Axes, drift: Axes, tau: float, dt: float,
                 turn_on_rng: np.random.Generator, rng: np.random.Generator) -> None:
        self.turn_on = _axes(turn_on) * turn_on_rng.standard_normal(3)
        self.drift = np.zeros(3)
        self._phi = math.exp(-dt / tau)
        # Keeps the drift's steady-state sigma at the given value.
        self._drift_step = _axes(drift) * math.sqrt(1.0 - self._phi ** 2)
        self.white = _axes(noise) / math.sqrt(2.0 * dt)
        self._rng = rng

    @property
    def bias(self) -> NDArray[np.float64]:
        return self.turn_on + self.drift

    def read(self, truth: ArrayLike) -> NDArray[np.float64]:
        # Six draws whatever the parameters, so a zero sigma does not shift the
        # stream.
        w = self._rng.standard_normal(6)
        self.drift = self._phi * self.drift + self._drift_step * w[:3]
        return np.asarray(truth, dtype=np.float64) + self.bias + self.white * w[3:]


class ImuModel:
    """The IMU errors of one run: :meth:`read` once per IMU frame, in order."""

    def __init__(self, spec: ImuSpec, dt: float, seed: int) -> None:
        if dt <= 0.0 or spec.drift_tau <= 0.0:
            raise ValueError("dt and drift_tau must be > 0")
        self.spec = spec
        self.seed = seed
        turn_on, accel, gyro = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
        )
        self.accel = _Triad(spec.accel_noise, spec.accel_turn_on, spec.accel_drift,
                            spec.drift_tau, dt, turn_on, accel)
        self.gyro = _Triad(spec.gyro_noise, spec.gyro_turn_on, spec.gyro_drift,
                           spec.drift_tau, dt, turn_on, gyro)

    def read(
        self, accel_frd: ArrayLike, gyro_frd: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The frame's measured specific force (m/s^2) and rate (rad/s), FRD,
        quantised for PX4's decode."""
        return (quantise(self.accel.read(accel_frd), ACCEL_LSB),
                quantise(self.gyro.read(gyro_frd), GYRO_LSB))

    def describe(self) -> str:
        a, g = self.accel, self.gyro
        return (
            f"IMU errors: {self.spec.name}, seed {self.seed}; turn-on bias FRD accel "
            f"[{', '.join(f'{v / _MG:+.2f}' for v in a.turn_on)}] mg, gyro "
            f"[{', '.join(f'{v / _DEG:+.3f}' for v in g.turn_on)}] deg/s; noise per "
            f"frame accel {a.white.max() / _MG:.2f} mg, gyro {g.white.max() / _DEG:.4f} deg/s"
        )


def build_imu(model: str, dt: float, seed: int) -> ImuModel | None:
    """The model ``--imu`` names, or None for ``ideal``."""
    if model not in IMU_MODELS:
        raise ValueError(f"imu model {model!r} not in {sorted(IMU_MODELS)}")
    spec = IMU_MODELS[model]
    return None if spec is None else ImuModel(spec, dt, seed)
