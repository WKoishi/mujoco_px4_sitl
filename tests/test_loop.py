"""Lockstep loop tests against a fake PX4. No PX4 build, no MuJoCo model.

These cover the two section 3.2 failure modes directly, because both are
invisible in a phase-1 smoke test: a loop that blocks per frame deadlocks at
startup, and a loop with no pacer runs away during PX4's boot window.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mavlink

from mujoco_px4_sitl import hil
from mujoco_px4_sitl.config import Config
from mujoco_px4_sitl.loop import LockstepLoop
from mujoco_px4_sitl.sim import StubPhysics
from mujoco_px4_sitl.transport import HilServer


class FakePX4:
    """A PX4 stand-in: TCP client that sends the two unsolicited startup
    messages and, optionally, HIL_ACTUATOR_CONTROLS in reply to HIL_SENSOR.
    """

    def __init__(self, port: int, *, reply: bool = True, lockstep: bool = True,
                 armed: bool = True) -> None:
        self.port = port
        self.reply = reply
        self.lockstep = lockstep
        self.armed = armed
        self.sensor_frames = 0
        self.state_frames = 0
        self.imu_timestamps: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        self._mav.robust_parsing = True

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _send_startup_messages(self, sock: socket.socket) -> None:
        """Order is not guaranteed in PX4 (two different threads), so send them
        in the awkward order: COMMAND_LONG first, then HEARTBEAT."""
        sock.sendall(mavlink.MAVLink_command_long_message(
            target_system=0, target_component=0,
            command=mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, confirmation=0,
            param1=float(mavlink.MAVLINK_MSG_ID_HIL_STATE_QUATERNION),
            param2=5000.0, param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0,
        ).pack(self._mav))
        sock.sendall(mavlink.MAVLink_heartbeat_message(
            type=mavlink.MAV_TYPE_QUADROTOR, autopilot=mavlink.MAV_AUTOPILOT_PX4,
            base_mode=0, custom_mode=0, system_status=mavlink.MAV_STATE_UNINIT,
            mavlink_version=3,
        ).pack(self._mav))

    def _actuator_message(self, time_usec: int) -> bytes:
        controls = [0.5] * 4 + [0.0] * (hil.NUM_ACTUATOR_OUTPUTS - 4)
        mode = hil.MODE_FLAG_CUSTOM | (hil.MODE_FLAG_ARMED if self.armed else 0)
        return mavlink.MAVLink_hil_actuator_controls_message(
            time_usec=time_usec, controls=controls, mode=mode,
            flags=hil.FLAG_LOCKSTEP if self.lockstep else 0,
        ).pack(self._mav)

    def _run(self) -> None:
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.2)
        self._send_startup_messages(sock)
        try:
            while not self._stop.is_set():
                try:
                    data = sock.recv(8192)
                except (TimeoutError, socket.timeout):
                    continue
                if not data:
                    return
                for msg in self._mav.parse_buffer(data) or []:
                    kind = msg.get_type()
                    if kind == "HIL_SENSOR":
                        self.sensor_frames += 1
                        self.imu_timestamps.append(int(msg.time_usec))
                        if self.reply:
                            sock.sendall(self._actuator_message(int(msg.time_usec)))
                    elif kind == "HIL_STATE_QUATERNION":
                        self.state_frames += 1
        finally:
            sock.close()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_loop(cfg: Config, *, reply: bool, lockstep: bool = True) -> tuple[LockstepLoop, FakePX4]:
    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    fake = FakePX4(cfg.hil_port, reply=reply, lockstep=lockstep)
    loop = LockstepLoop(cfg, StubPhysics(cfg), server, None)
    try:
        fake.start()
        loop.run()
    finally:
        fake.stop()
        server.close()
    return loop, fake


def make_config(**overrides) -> Config:
    cfg = Config(
        stub_physics=True, sidechannel_enabled=False, speed_factor=50.0,
        max_sim_time=0.4, brake_timeout_s=0.2, status_interval_s=1e9,
    )
    cfg.hil_port_base = free_port()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# --- the happy path -------------------------------------------------------

def test_loop_exchanges_both_messages_every_frame():
    """HIL_STATE_QUATERNION goes out on every IMU frame, i.e. 250 Hz -- above the
    200 Hz PX4 requests, which is not an integer divisor of the IMU rate.

    The two counts are compared with a tolerance of one message, not for equality.
    The pair is two separate ``sendall`` calls, so a teardown landing between them
    leaves the fake having read the sensor frame but not its state frame. That is a
    shutdown artifact of this harness; the loop itself sends both unconditionally.
    """
    cfg = make_config()
    loop, fake = run_loop(cfg, reply=True)
    expected = int(cfg.max_sim_time * cfg.imu_rate_hz)
    # Frame count is driven by simulated time, so it is deterministic apart from
    # the same teardown boundary; allow a few frames rather than a percentage.
    assert abs(loop.stats.frames - expected) <= 5
    assert abs(fake.state_frames - fake.sensor_frames) <= 1
    assert fake.sensor_frames > 0
    assert loop.stats.actuator_messages > 0


def test_startup_messages_are_discarded_without_breaking_framing():
    """PX4's unsolicited HEARTBEAT and COMMAND_LONG must not desynchronise the
    parser, and nothing may be gated on either arriving."""
    loop, fake = run_loop(make_config(), reply=True)
    assert loop.stats.discarded_messages >= 2
    assert loop.stats.actuator_messages > 0
    assert fake.sensor_frames > 0


def test_imu_timestamps_are_strictly_monotonic_with_stable_dt():
    """Our IMU timestamp *is* PX4's clock, so this is a correctness requirement."""
    cfg = make_config()
    _, fake = run_loop(cfg, reply=True)
    stamps = np.asarray(fake.imu_timestamps, dtype=np.int64)
    assert len(stamps) > 10
    deltas = np.diff(stamps)
    assert np.all(deltas > 0)
    assert np.all(deltas == pytest.approx(1e6 / cfg.imu_rate_hz, abs=1))


def test_lockstep_flag_absence_is_survivable():
    """A nolockstep build must produce a warning, not a crash."""
    loop, _ = run_loop(make_config(), reply=True, lockstep=False)
    assert loop.stats.actuator_messages > 0
    assert not loop.controls.lockstep


# --- the section 3.2 failure modes ---------------------------------------

def test_silent_px4_does_not_deadlock_the_loop():
    """The critical test. PX4 publishes no HIL_ACTUATOR_CONTROLS before Commander
    is up, and its escape hatches are either measured in simulated time or
    compiled out under lockstep. A loop that waits per frame hangs forever here.
    """
    cfg = make_config(max_sim_time=0.2, brake_timeout_s=0.05)
    loop, fake = run_loop(cfg, reply=False)
    assert loop.stats.frames == pytest.approx(int(0.2 * cfg.imu_rate_hz), rel=0.05)
    assert loop.stats.actuator_messages == 0
    assert loop.stats.brake_timeouts > 0
    # Timing out must hand pacing back to the wall clock, not stall the loop.
    assert fake.sensor_frames > 0


def test_brake_bounds_the_lead_over_a_silent_px4():
    """Timeouts must not exceed one per ``max_lead_frames`` frames, or the brake
    is pacing the loop instead of the pacer -- the ~100x silent slowdown."""
    cfg = make_config(max_sim_time=0.2, brake_timeout_s=0.02, max_lead_frames=8)
    loop, _ = run_loop(cfg, reply=False)
    assert loop.stats.brake_timeouts <= loop.stats.frames // cfg.max_lead_frames + 2


def test_pacer_bounds_the_rate_even_when_px4_is_silent():
    """Without a pacer, a silent PX4 lets simulated time run away -- which is
    exactly PX4's boot window, where nothing else bounds us.
    """
    cfg = make_config(max_sim_time=0.5, speed_factor=5.0, brake_timeout_s=0.02)
    start = time.monotonic()
    loop, _ = run_loop(cfg, reply=False)
    elapsed = time.monotonic() - start
    # 0.5 s of simulated time at 5x may not arrive faster than 0.1 s of wall
    # clock. A free-running loop would finish in single-digit milliseconds.
    assert elapsed >= 0.5 / cfg.speed_factor * 0.8
    assert loop.stats.frames > 0


def test_speed_factor_governs_the_sim_to_wall_ratio():
    """The section 7 diagnostic: the ratio must track ``speed_factor``, measured
    inside the loop so connection setup is not counted.

    The bounds are deliberately asymmetric. Exceeding ``speed_factor`` is the
    dangerous direction -- it means we outran PX4 and IMU FIFO samples are being
    dropped -- so it is held tight. Falling short only needs to show the ratio has
    not collapsed; a loose bound keeps this from being a machine-speed test, since
    a single brake timeout costs real wall clock.
    """
    cfg = make_config(max_sim_time=0.6, speed_factor=2.0)
    loop, _ = run_loop(cfg, reply=True)
    assert loop.stats.ratio <= cfg.speed_factor * 1.1
    assert loop.stats.ratio >= cfg.speed_factor * 0.4


def test_ratio_does_not_run_away_when_px4_is_silent():
    """A ratio that climbs without bound is the "we outran PX4" fault, and PX4's
    boot window is exactly a silent PX4."""
    cfg = make_config(max_sim_time=0.3, speed_factor=2.0, brake_timeout_s=0.02)
    loop, _ = run_loop(cfg, reply=False)
    assert loop.stats.ratio <= cfg.speed_factor * 1.5


def test_disarmed_px4_yields_zero_controls():
    cfg = make_config(max_sim_time=0.2)
    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    fake = FakePX4(cfg.hil_port, reply=True, armed=False)
    loop = LockstepLoop(cfg, StubPhysics(cfg), server, None)
    try:
        fake.start()
        loop.run()
    finally:
        fake.stop()
        server.close()
    assert loop.stats.actuator_messages > 0
    assert not loop.controls.armed
    assert loop.controls.effective(4) == pytest.approx(np.zeros(4))
