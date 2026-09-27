"""IMU error model tests. No PX4 build, no MuJoCo model load.

The statistics are checked against the spec they were built from, so a units
slip (deg for rad, mg for m/s^2, a sqrt(2) in the white noise) fails here rather
than as an estimator that looks a little too good or a little too bad.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mavlink

from mujoco_px4_sitl import frames
from mujoco_px4_sitl.config import Config, config_from_args
from mujoco_px4_sitl.loop import LockstepLoop
from mujoco_px4_sitl.sensors import (
    ACCEL_LSB, GYRO_LSB, ICM_42688_P, ImuModel, ImuSpec, build_imu, quantise,
)
from mujoco_px4_sitl.sim import StubPhysics

DT = 0.004
QUIET = ImuSpec(name="quiet", gyro_noise=0.0, gyro_turn_on=0.0, gyro_drift=0.0,
                accel_noise=0.0, accel_turn_on=0.0, accel_drift=0.0, drift_tau=300.0)


def readings(model: ImuModel, frames_: int, accel=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
    out = [model.read(accel, gyro) for _ in range(frames_)]
    return np.array([a for a, _ in out]), np.array([g for _, g in out])


def errors(model: ImuModel, frames_: int):
    """Before quantisation, where the statistics are the spec's."""
    out = [(model.accel.read(np.zeros(3)), model.gyro.read(np.zeros(3)))
           for _ in range(frames_)]
    return np.array([a for a, _ in out]), np.array([g for _, g in out])


def px4_decode(value, lsb: float) -> np.ndarray:
    """SimulatorMavlink.cpp:222-224: float32 over the scale, into int16, which
    truncates toward zero."""
    return np.trunc(np.asarray(value, dtype=np.float32) / np.float32(lsb))


def test_the_same_seed_repeats_frame_for_frame_and_another_does_not():
    a = readings(ImuModel(ICM_42688_P, DT, seed=7), 50)
    b = readings(ImuModel(ICM_42688_P, DT, seed=7), 50)
    c = readings(ImuModel(ICM_42688_P, DT, seed=8), 50)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_white_noise_per_frame_is_the_density_over_half_the_frame_rate():
    """0.0028 deg/s/sqrt(Hz) at 250 Hz is 0.031 deg/s per frame, not the 0.044
    that density / sqrt(dt) gives."""
    spec = replace(QUIET, gyro_noise=ICM_42688_P.gyro_noise,
                   accel_noise=ICM_42688_P.accel_noise)
    accel, gyro = errors(ImuModel(spec, DT, seed=1), 20000)
    want_gyro = ICM_42688_P.gyro_noise * math.sqrt(0.5 / DT)
    assert math.degrees(want_gyro) == pytest.approx(0.0313, abs=1e-4)
    assert gyro.std(axis=0) == pytest.approx([want_gyro] * 3, rel=0.03)
    want_accel = np.array(ICM_42688_P.accel_noise) * math.sqrt(0.5 / DT)
    assert accel.std(axis=0) == pytest.approx(want_accel, rel=0.03)
    assert np.abs(gyro.mean(axis=0)).max() < 4 * want_gyro / math.sqrt(20000)


def test_px4s_truncation_decodes_the_rounded_count():
    """Sent as is, a value PX4 truncates loses everything under one LSB: the
    gyro's noise per frame is half an LSB, so at rest most of it would vanish,
    and a bias would read half an LSB small."""
    x = np.array([0.0, 0.3, 0.49, 0.51, 1.2, -0.3, -0.51, -1.7, 2047.6, -2048.4]) * GYRO_LSB
    assert np.array_equal(px4_decode(quantise(x, GYRO_LSB), GYRO_LSB), np.round(x / GYRO_LSB))
    g = np.array([0.0, 0.0, -1.0]) * frames.STANDARD_GRAVITY
    assert np.array_equal(px4_decode(quantise(g, ACCEL_LSB), ACCEL_LSB), [0, 0, -2048])


def test_quantisation_saturates_at_full_scale():
    """PX4's int16 cast does not saturate: past full scale it wraps, a sign flip."""
    big = np.array([40.0, -40.0, 0.0])  # rad/s, past 2000 deg/s
    assert np.array_equal(px4_decode(quantise(big, GYRO_LSB), GYRO_LSB), [32767, -32767, 0])


def test_the_noise_survives_px4s_decode():
    """At rest, with ICM-42688-P noise and turn-on bias (no drift, which would
    move the centre): the decoded counts spread like the noise, plus the
    rounding's LSB^2 / 12, and centre on the bias."""
    model = ImuModel(replace(ICM_42688_P, gyro_drift=0.0), DT, seed=4)
    _, gyro = readings(model, 20000)
    counts = px4_decode(gyro, GYRO_LSB)
    sigma = ICM_42688_P.gyro_noise * math.sqrt(0.5 / DT) / GYRO_LSB
    assert counts.std(axis=0) == pytest.approx([math.hypot(sigma, math.sqrt(1 / 12))] * 3,
                                               rel=0.05)
    assert counts.mean(axis=0) == pytest.approx(model.gyro.bias / GYRO_LSB, abs=0.05)


def test_the_turn_on_bias_is_constant_and_drawn_at_its_sigma():
    spec = replace(QUIET, gyro_turn_on=0.01, accel_turn_on=(0.1, 0.2, 0.3))
    model = ImuModel(spec, DT, seed=3)
    accel, gyro = readings(model, 20)
    assert np.all(accel == quantise(model.accel.turn_on, ACCEL_LSB))
    assert np.all(gyro == quantise(model.gyro.turn_on, GYRO_LSB))
    draws = np.array([ImuModel(spec, DT, seed=s).accel.turn_on for s in range(4000)])
    assert draws.std(axis=0) == pytest.approx([0.1, 0.2, 0.3], rel=0.05)
    assert np.abs(draws.mean(axis=0)).max() < 0.02


def test_drift_is_gauss_markov_at_its_steady_state_sigma_and_time_constant():
    tau = 10 * DT
    spec = replace(QUIET, gyro_drift=0.02, drift_tau=tau)
    _, gyro = errors(ImuModel(spec, DT, seed=5), 40000)
    x = gyro[1000:, 0]
    assert x.std() == pytest.approx(0.02, rel=0.05)
    lag = 10
    rho = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    assert rho == pytest.approx(math.exp(-lag * DT / tau), abs=0.03)


def test_drift_starts_at_zero():
    spec = replace(QUIET, gyro_drift=0.02)
    model = ImuModel(spec, DT, seed=5)
    _, gyro = errors(model, 1)
    # One step of a 300 s process: a small fraction of its sigma.
    assert np.abs(gyro).max() < 0.02 * 0.02


def test_changing_the_gyro_does_not_shift_the_accel_stream():
    """Separate streams: a gyro parameter study keeps every accel draw."""
    a, _ = errors(ImuModel(ICM_42688_P, DT, seed=2), 30)
    louder = replace(ICM_42688_P, gyro_noise=1.0, gyro_turn_on=0.0, gyro_drift=0.0)
    b, _ = errors(ImuModel(louder, DT, seed=2), 30)
    assert np.array_equal(a, b)


def test_the_icm42688p_budget_stays_clear_of_px4s_arming_and_rest_checks():
    """Arming fails at an accel bias of 0.75 * EKF2_ABL_LIM = 0.3 m/s^2 plus 3
    sigma (estimatorCheck.cpp:490-497); the land detector leaves rest at a
    vibration metric of 0.02 rad/s (LandDetector.cpp:246-256), which a gyro
    sigma of about 0.025 rad/s per frame reaches. A 3 sigma draw of both biases
    stays under half the first; the white noise, a twentieth of the second."""
    spec = ICM_42688_P
    assert 3 * spec.accel_turn_on + 3 * spec.accel_drift < 0.3 / 2
    assert spec.gyro_noise * math.sqrt(0.5 / DT) < 0.025 / 20


def test_ideal_builds_nothing():
    assert build_imu("ideal", DT, 0) is None
    with pytest.raises(ValueError, match="not in"):
        build_imu("bmi088", DT, 0)


def test_config_takes_the_model_and_the_seed():
    cfg = config_from_args(["--stub-physics", "--imu", "ideal", "--imu-seed", "4"])
    assert (cfg.imu_model, cfg.imu_seed) == ("ideal", 4)
    assert config_from_args(["--stub-physics"]).imu_model == "icm42688p"
    with pytest.raises(ValueError, match="imu_model"):
        Config(stub_physics=True, imu_model="bmi088").validate()


# --- on the wire -----------------------------------------------------------


class Capture:
    """A HIL server stand-in that keeps what the loop sends; PX4 stays silent."""

    connected = True

    def __init__(self) -> None:
        self.mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        self.sent: list[bytes] = []

    def send(self, payload: bytes) -> bool:
        self.sent.append(payload)
        return True

    def drain(self):
        return iter(())

    def wait(self, timeout: float):  # noqa: ARG002
        return iter(())

    def messages(self, kind: str) -> list:
        parser = mavlink.MAVLink(None)
        out = []
        for payload in self.sent:
            out += [m for m in parser.parse_buffer(payload) or [] if m.get_type() == kind]
        return out


def fly_stub(**overrides) -> tuple[Config, Capture]:
    cfg = Config(stub_physics=True, sidechannel_enabled=False, speed_factor=50.0,
                 max_sim_time=0.1, status_interval_s=1e9)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    server = Capture()
    LockstepLoop(cfg, StubPhysics(cfg), server, None).run()
    return cfg, server


def test_only_hil_sensor_carries_the_errors():
    """The stub is level and still: HIL_STATE_QUATERNION must say exactly that,
    and HIL_SENSOR must carry the seed's errors, frame for frame."""
    cfg, server = fly_stub(imu_seed=11)
    sensors = server.messages("HIL_SENSOR")
    states = server.messages("HIL_STATE_QUATERNION")
    assert len(sensors) == len(states) >= 20
    for m in states:
        assert (m.rollspeed, m.pitchspeed, m.yawspeed) == (0.0, 0.0, 0.0)
        assert (m.xacc, m.yacc, m.zacc) == (0, 0, -9807)
    replay = ImuModel(ICM_42688_P, cfg.imu_dt, seed=11)
    truth = np.array([0.0, 0.0, -frames.STANDARD_GRAVITY])
    for m in sensors:
        accel, gyro = replay.read(truth, np.zeros(3))
        assert [m.xacc, m.yacc, m.zacc] == pytest.approx(accel, abs=1e-6)
        assert [m.xgyro, m.ygyro, m.zgyro] == pytest.approx(gyro, abs=1e-7)


def test_ideal_sends_the_truth():
    _, server = fly_stub(imu_model="ideal")
    for m in server.messages("HIL_SENSOR"):
        assert (m.xgyro, m.ygyro, m.zgyro) == (0.0, 0.0, 0.0)
        # Not quantised either: ideal is the truth, as flown before.
        assert m.zacc == pytest.approx(-frames.STANDARD_GRAVITY, abs=1e-6)
