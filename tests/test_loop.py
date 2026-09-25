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
from mujoco_px4_sitl.sidechannel import SCHEMA_VERSION, SideChannel
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


class ScriptedServer:
    """A :class:`HilServer` stand-in whose inbound messages we control exactly.

    Timing-free, so the lead-counter invariant can be asserted directly rather
    than inferred from how often the brake happened to fire.
    """

    connected = True

    def __init__(self, queued: list[mavlink.MAVLink_message] | None = None) -> None:
        self.mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        self.queued = list(queued or [])
        self.sent: list[bytes] = []

    def send(self, payload: bytes) -> bool:
        self.sent.append(payload)
        return True

    def drain(self):
        pending, self.queued = self.queued, []
        yield from pending

    def wait(self, timeout: float):  # noqa: ARG002 - never reached in these tests
        yield from self.drain()


def an_actuator_message(*, armed: bool = True) -> mavlink.MAVLink_message:
    """A HIL_ACTUATOR_CONTROLS as the loop's inbound path sees it (unpacked)."""
    controls = [0.5] * 4 + [0.0] * (hil.NUM_ACTUATOR_OUTPUTS - 4)
    mode = hil.MODE_FLAG_CUSTOM | (hil.MODE_FLAG_ARMED if armed else 0)
    return mavlink.MAVLink_hil_actuator_controls_message(
        time_usec=1, controls=controls, mode=mode, flags=hil.FLAG_LOCKSTEP,
    )


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


# --- the lead counter, which is what makes the brake a brake ---------------

def test_draining_a_fresh_actuator_message_clears_the_lead():
    """The non-blocking drain path must reset ``frames_since_ack``.

    If only the brake resets it, braking degenerates into a timer that fires
    every ``max_lead_frames`` frames however promptly PX4 replies -- plan 7's
    "the brake is pacing the loop" fault. At ``speed_factor = 1.0`` the pacer's
    own sleep hides the cost, so nothing else catches this.
    """
    cfg = make_config()
    loop = LockstepLoop(cfg, StubPhysics(cfg), ScriptedServer([an_actuator_message()]))
    loop._frames_since_ack = 7

    loop._drain()

    assert loop.stats.actuator_messages == 1
    assert loop._frames_since_ack == 0


def test_draining_other_messages_leaves_the_lead_alone():
    """Only HIL_ACTUATOR_CONTROLS clears it. PX4's unsolicited HEARTBEAT and
    COMMAND_LONG say nothing about whether its control chain has caught up."""
    cfg = make_config()
    heartbeat = mavlink.MAVLink_heartbeat_message(
        type=mavlink.MAV_TYPE_QUADROTOR, autopilot=mavlink.MAV_AUTOPILOT_PX4,
        base_mode=0, custom_mode=0, system_status=mavlink.MAV_STATE_UNINIT,
        mavlink_version=3,
    )
    loop = LockstepLoop(cfg, StubPhysics(cfg), ScriptedServer([heartbeat]))
    loop._frames_since_ack = 7

    loop._drain()

    assert loop.stats.discarded_messages == 1
    assert loop._frames_since_ack == 7


def test_a_responsive_px4_is_never_braked():
    """End to end: PX4 replying to every frame must not brake at all.

    The bug this pins down reported ``brake == frames / max_lead_frames`` with a
    PX4 that answered every single frame. ``max_lead_frames`` is small and the
    speed factor modest so the fake has room to keep up; the bound still leaves a
    wide margin against the ~30 brake waits the fault produced here.
    """
    cfg = make_config(max_sim_time=0.3, speed_factor=5.0, max_lead_frames=8)
    loop, fake = run_loop(cfg, reply=True)
    assert loop.stats.actuator_messages > 0
    assert loop.stats.brake_waits <= 2, (
        f"braked {loop.stats.brake_waits} times against a PX4 that answered "
        f"{loop.stats.actuator_messages} of {loop.stats.frames} frames"
    )
    assert loop.stats.brake_timeouts == 0


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


# --- send back-pressure ---------------------------------------------------

def test_a_full_send_buffer_does_not_drop_a_live_px4():
    """A PX4 that stops draining its receive buffer is slow, not gone.

    The socket is non-blocking, so a full buffer raises ``BlockingIOError``,
    which is an ``OSError`` -- catching it alongside the real disconnects tears
    down a live link, and ``sendall`` may write part of a frame before raising,
    silently desynchronising PX4's parser (plan 3.1). Neither may happen.

    Buffers are pinned on both ends so the stall is deterministic: an explicit
    ``SO_RCVBUF`` also disables Linux's receive-window autotuning, which could
    otherwise absorb the whole payload.
    """
    port = free_port()
    server = HilServer("127.0.0.1", port)
    client = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    try:
        assert server.accept(timeout=2.0)
        server._conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

        # Larger than both buffers together, sent as one message. The byte
        # pattern makes a partial or reordered write visible.
        payload = bytes(range(256)) * 4096  # 1 MiB
        result: list[bool] = []
        writer = threading.Thread(target=lambda: result.append(server.send(payload)))
        writer.start()

        # While the client refuses to read, the write must be pending -- not
        # failed, and the link must still be up. This is the assertion the old
        # code failed: it dropped the client here.
        time.sleep(0.4)
        assert writer.is_alive(), "send() returned without the payload draining"
        assert server.connected, "a slow reader was treated as a disconnect"

        received = bytearray()
        client.settimeout(20.0)
        while len(received) < len(payload):
            chunk = client.recv(1 << 20)
            if not chunk:
                break
            received.extend(chunk)
        writer.join(timeout=20.0)

        assert not writer.is_alive(), "send() never returned"
        assert result == [True]
        assert server.connected
        # The whole frame arrived in order: no partial write was abandoned.
        assert bytes(received) == payload
        assert server.send_stalls == 1
    finally:
        client.close()
        server.close()


def test_send_still_reports_a_real_disconnect():
    """The back-pressure path must not swallow an actual peer loss."""
    port = free_port()
    server = HilServer("127.0.0.1", port)
    client = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        assert server.accept(timeout=2.0)
        client.close()
        # The first write may land in the buffer before RST arrives; the loop
        # treats any False as a lost link, so a bounded retry is faithful to it.
        for _ in range(200):
            if not server.send(b"x" * 4096):
                break
            time.sleep(0.005)
        assert not server.connected
    finally:
        client.close()
        server.close()


# --- shutdown -------------------------------------------------------------

def test_stop_interrupts_the_wait_for_px4():
    """PX4 may never connect -- a wrong airframe id is enough. A wait that
    ignores stop() hangs run_sitl.sh's cleanup, which signals and then waits.
    """
    cfg = make_config()
    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    loop = LockstepLoop(cfg, StubPhysics(cfg), server, None)
    try:
        runner = threading.Thread(target=loop.run)
        runner.start()
        time.sleep(0.3)  # let it reach the accept loop
        loop.stop()
        runner.join(timeout=5.0)
        assert not runner.is_alive(), "stop() did not interrupt the accept loop"
        assert loop.stats.frames == 0
    finally:
        server.close()


# --- the side channel's arm path ------------------------------------------


class RecordingPhysics(StubPhysics):
    """The stub, noting when arm commands arrive relative to frame steps."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.events: list[tuple[str, float]] = []

    def submit_arm_command(self, command) -> None:
        self.events.append(("arm_cmd", self.time))

    def step_frame(self, controls) -> None:
        self.events.append(("step", self.time))
        super().step_frame(controls)


def test_arm_cmds_are_taken_every_frame_not_at_the_publish_rate():
    """Ground truth goes out every fifth frame; an arm_cmd must not wait for it.

    Polling only when publishing would hold a command for up to 20 ms -- a delay
    no controller asked for, and one the research could not tell from its own.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    cfg = make_config(sidechannel_enabled=True, max_sim_time=0.1)
    assert cfg.imu_rate_hz / cfg.sidechannel_rate_hz == 5
    channel = SideChannel("127.0.0.1", port)
    physics = RecordingPhysics(cfg)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.sendto(
                b'{"v": %d, "type": "arm_cmd", "values": [0.1]}' % SCHEMA_VERSION,
                ("127.0.0.1", port),
            )
            deadline = time.monotonic() + 1.0
            while not select_readable(channel.sock) and time.monotonic() < deadline:
                time.sleep(0.001)
            LockstepLoop(cfg, physics, ScriptedServer(), channel).run()
    finally:
        channel.close()
    assert physics.events[:2] == [("arm_cmd", 0.0), ("step", 0.0)]


def select_readable(sock: socket.socket) -> bool:
    import select

    return bool(select.select([sock], [], [], 0.0)[0])
