"""The in-process controller host, its schedule, and PX4's estimate link.

MuJoCo only for the host (a welded base and a two-joint arm, as in
``test_arm.py``), a fake PX4 over UDP for the estimate link, and the stub
physics for the loop. No PX4 build, no private geometry.

What matters most here is timing that would otherwise be silent: a command that
reaches the servos a frame early or late, a drop schedule that loses more in a
row than it promised, an age bound that the arm's own watchdog does not honour,
and an out-of-order estimate taken for the newest.
"""

from __future__ import annotations

import logging
import math
import socket
import time

import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mavlink

from mujoco_px4_sitl import frames, px4link
from mujoco_px4_sitl.config import Config
from mujoco_px4_sitl.control import (
    CappedDrops,
    ControllerHost,
    Observation,
    Sample,
    Schedule,
)
from mujoco_px4_sitl.loop import LockstepLoop
from mujoco_px4_sitl.main import run
from mujoco_px4_sitl.px4link import EstimateLink
from mujoco_px4_sitl.sim import MujocoPhysics, StubPhysics
from mujoco_px4_sitl.transport import HilServer

IMU_DT = 1.0 / 250.0

_ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="base_link" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.02" mass="2.0" contype="0" conaffinity="0"/>
      <site name="imu"/>
      <site name="rotor0" pos="0.2 -0.2 0"/>
      <site name="rotor1" pos="-0.2 0.2 0"/>
      <site name="rotor2" pos="0.2 0.2 0"/>
      <site name="rotor3" pos="-0.2 -0.2 0"/>
      <body name="arm_link0" pos="0 0 -0.05">
        <joint name="arm_joint0" axis="0 0 1" range="-3 3" damping="0.5" armature="0.01"/>
        <geom type="sphere" size="0.01" mass="0.05" contype="0" conaffinity="0"/>
        <body name="arm_link1">
          <joint name="arm_joint1" axis="0 1 0" range="-1.5 1.5" damping="0.3"
                 armature="0.005"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld body1="base_link"/>
  </equality>
  <actuator>
    <position name="arm_act0" joint="arm_joint0" kp="20" kv="1" ctrlrange="-3 3"/>
    <position name="arm_act1" joint="arm_joint1" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
  </actuator>
  <sensor>
    <accelerometer name="imu_accel" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def sim(tmp_path) -> MujocoPhysics:
    path = tmp_path / "arm.xml"
    path.write_text(_ARM_XML)
    return MujocoPhysics(Config(model_path=path, arm_timeout_s=0.0))


class Scripted:
    """A controller that returns ``target`` (or what ``fn`` says) and keeps
    every observation it was given."""

    def __init__(self, target=(0.3, 0.2), fn=None) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.fn = fn
        self.seen: list[Observation] = []

    def step(self, obs: Observation):
        self.seen.append(obs)
        return self.fn(obs) if self.fn is not None else self.target


def drive(host: ControllerHost, sim, frames: int, each=None) -> None:
    """The loop's order: host, then step. ``each(frame)`` runs after the host."""
    for frame in range(frames):
        host.before_step(sim)
        if each is not None:
            each(frame)
        sim.step_frame(np.zeros(4))


# --- the schedule ---------------------------------------------------------


def test_the_controller_runs_every_period_of_simulated_time(sim):
    controller = Scripted()
    host = ControllerHost(controller, Schedule(period=0.02), IMU_DT)
    drive(host, sim, 50)
    assert [o.index for o in controller.seen] == list(range(10))
    assert [o.time for o in controller.seen] == pytest.approx([0.02 * k for k in range(10)])


@pytest.mark.parametrize("delay_frames", [0, 1, 3, 7])
def test_a_command_reaches_the_servos_after_exactly_the_delay(sim, delay_frames):
    """Delay 0 drives the frame that starts at the sample instant itself; a
    delay longer than the period leaves several commands in flight."""
    host = ControllerHost(
        Scripted(fn=lambda o: [0.1 * (o.index + 1), 0.0]),
        Schedule(period=0.02, delay=delay_frames * IMU_DT), IMU_DT,
    )
    arrivals: dict[int, int] = {}

    def watch(frame):
        command = sim.arm.command
        if command is not None and command.seq not in arrivals:
            arrivals[command.seq] = frame
            # The servos see the sample instant as the command's data time.
            assert command.state_time == pytest.approx(command.seq * 0.02)

    drive(host, sim, 60, watch)
    assert arrivals == {k: 5 * k + delay_frames for k in range(len(arrivals))}
    assert len(arrivals) >= 10


def test_none_sends_nothing(sim):
    host = ControllerHost(Scripted(fn=lambda o: None), Schedule(period=0.02), IMU_DT)
    drive(host, sim, 30)
    assert sim.arm.command is None
    assert (host.samples, host.commands, host.idle) == (6, 0, 6)


def test_command_age_reaches_but_never_exceeds_the_schedule_bound(sim):
    """The point of the whole module: H <= (N + 1) h + tau, known in advance.

    Read at frame starts, cmd_age peaks one frame below the continuous-time
    bound, and it must get there -- a bound never approached could be loose by
    any amount without this test noticing.
    """
    schedule = Schedule(period=0.02, delay=0.008, drops=CappedDrops(0.5, 2, seed=3))
    host = ControllerHost(Scripted(), schedule, IMU_DT)
    ages = []
    drive(host, sim, 1000, lambda f: ages.append(sim.arm.age(sim.time)))
    ages = np.array([a for a in ages if a is not None])
    assert schedule.age_bound() == pytest.approx(3 * 0.02 + 0.008)
    assert ages.max() <= schedule.age_bound() - IMU_DT + 1e-9
    assert ages.max() == pytest.approx(schedule.age_bound() - IMU_DT)
    assert host.dropped > 0


def test_capped_drops_keep_their_cap_and_repeat_with_their_seed():
    def pattern(seed):
        drops = CappedDrops(0.7, 3, seed=seed)
        return [drops(k) for k in range(2000)]

    a = pattern(11)
    assert a == pattern(11)
    assert a != pattern(12)
    longest = run_length = 0
    for dropped in a:
        run_length = run_length + 1 if dropped else 0
        longest = max(longest, run_length)
    assert longest == 3


def test_the_draws_do_not_depend_on_the_cap():
    """Raising the cap changes only the samples the cap overrode."""
    low = CappedDrops(0.6, 1, seed=5)
    high = CappedDrops(0.6, 1000, seed=5)
    a = [low(k) for k in range(500)]
    b = [high(k) for k in range(500)]
    assert all(y for x, y in zip(a, b) if x)  # every low drop is a high drop


def test_an_uncapped_drop_schedule_states_no_bound():
    assert Schedule(period=0.02, drops=lambda k: k % 7 == 0).age_bound() is None
    assert Schedule(period=0.02).age_bound() == pytest.approx(0.02)


@pytest.mark.parametrize("kwargs", [
    {"period": 0.003}, {"period": 0.0}, {"period": 0.02, "delay": -0.004},
    {"period": 0.02, "delay": 0.005},
])
def test_a_schedule_off_the_imu_grid_is_refused(kwargs):
    with pytest.raises(ValueError, match="IMU frames"):
        Schedule(**kwargs).frames(IMU_DT)


def test_a_watchdog_inside_the_age_bound_is_called_out(caplog):
    host = ControllerHost(
        Scripted(), Schedule(period=0.02, drops=CappedDrops(0.1, 4)), IMU_DT
    )
    with caplog.at_level(logging.WARNING):
        line = host.describe(arm_timeout=0.05)
    assert "below the schedule's command-age bound" in caplog.text
    assert "drops <= 4 in a row" in line and "0.1 s" in line


# --- what the controller sees, and what only the recorder sees -------------


def test_joints_are_read_at_the_sample_instant_and_truth_goes_to_the_recorder(sim):
    samples: list[Sample] = []
    controller = Scripted(target=(0.5, -0.3))
    host = ControllerHost(controller, Schedule(period=0.02), IMU_DT, recorder=samples.append)
    drive(host, sim, 100)

    last = samples[-1]
    joints = sim.model.actuator_trnid[sim.arm.ids, 0]
    assert last.obs is controller.seen[-1]
    assert last.obs.estimate is None and last.obs.estimate_age is None
    # Ideal encoders: the reading is the joint state at t_k, which truth repeats.
    assert last.obs.joints.q == pytest.approx(last.truth.joints.q)
    assert last.truth.state.time == pytest.approx(last.obs.time)
    assert last.truth.arm is not None and last.truth.pos_ned_ekf is None
    assert last.apply_time == pytest.approx(last.obs.time)
    # By now the servos have pulled the joints most of the way there.
    q = np.array([sim.data.qpos[sim.model.jnt_qposadr[j]] for j in joints])
    assert sim.arm_joints().q == pytest.approx(q)
    assert q == pytest.approx([0.5, -0.3], abs=0.1)


# --- the estimate link ----------------------------------------------------


class FakeApiLink:
    """PX4's end of the API link: a UDP socket sending MAVLink 2 to ours."""

    def __init__(self, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(1.0)
        self.target = ("127.0.0.1", port)
        self.mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        self.mav.robust_parsing = True

    def send(self, msg) -> None:
        self.sock.sendto(msg.pack(self.mav), self.target)

    def heartbeat(self) -> None:
        self.send(mavlink.MAVLink_heartbeat_message(
            type=mavlink.MAV_TYPE_OCTOROTOR, autopilot=mavlink.MAV_AUTOPILOT_PX4,
            base_mode=0, custom_mode=0, system_status=mavlink.MAV_STATE_STANDBY,
            mavlink_version=3,
        ))

    def odometry(self, time_usec: int, x: float = 1.0, frame=mavlink.MAV_FRAME_LOCAL_NED,
                 child=mavlink.MAV_FRAME_LOCAL_NED, reset: int = 0) -> None:
        self.send(mavlink.MAVLink_odometry_message(
            time_usec=time_usec, frame_id=frame, child_frame_id=child,
            x=x, y=2.0, z=-3.0, q=[1.0, 0.0, 0.0, 0.0], vx=0.1, vy=0.2, vz=0.3,
            rollspeed=0.01, pitchspeed=0.02, yawspeed=0.03,
            pose_covariance=[math.nan] * 21, velocity_covariance=[math.nan] * 21,
            reset_counter=reset, estimator_type=mavlink.MAV_ESTIMATOR_TYPE_AUTOPILOT,
            quality=100,
        ))

    def received(self) -> list:
        out = []
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                out.extend(self.mav.parse_buffer(data) or [])
        except (TimeoutError, socket.timeout):
            return out

    def close(self) -> None:
        self.sock.close()


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def api():
    port = _free_udp_port()
    link = EstimateLink("127.0.0.1", port, rate_hz=250.0)
    fake = FakeApiLink(port)
    yield link, fake
    fake.close()
    link.close()


def _poll_until(link: EstimateLink, now: float, done, timeout: float = 2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        link.poll(now)
        if done():
            return
        time.sleep(0.005)
    raise AssertionError("estimate link never got there")


def test_the_newest_estimate_wins_and_a_late_older_one_is_ignored(api):
    link, fake = api
    fake.odometry(1000, x=1.0)
    fake.odometry(3000, x=3.0)
    fake.odometry(2000, x=2.0)  # reordered on the way
    _poll_until(link, 0.005, lambda: link.received + link.out_of_order == 3)

    est = link.latest
    assert est.time == pytest.approx(0.003) and est.pos_ned[0] == 3.0
    assert est.received == pytest.approx(0.005)
    assert est.vel_ned == pytest.approx([0.1, 0.2, 0.3])
    assert est.rates_frd == pytest.approx([0.01, 0.02, 0.03])
    assert (link.received, link.out_of_order) == (2, 1)
    assert link.max_delivery == pytest.approx(0.005 - 0.001)


def test_odometry_not_in_local_ned_is_refused_not_rotated(api):
    link, fake = api
    fake.odometry(1000, child=mavlink.MAV_FRAME_BODY_FRD)
    _poll_until(link, 0.0, lambda: link.rejected == 1)
    assert link.latest is None


def test_the_rate_is_asked_for_after_the_heartbeat_until_acknowledged(api, monkeypatch):
    link, fake = api
    monkeypatch.setattr(px4link, "_REQUEST_RETRY_S", 0.0)
    link.poll(0.0)
    assert fake.received() == []  # nowhere to send it before PX4 is heard

    fake.heartbeat()
    _poll_until(link, 0.0, lambda: link._target is not None)
    requests = [m for m in fake.received() if m.get_type() == "COMMAND_LONG"]
    assert requests, "no SET_MESSAGE_INTERVAL after the heartbeat"
    assert requests[0].command == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert requests[0].param1 == mavlink.MAVLINK_MSG_ID_ODOMETRY
    assert requests[0].param2 == pytest.approx(4000.0)

    fake.send(mavlink.MAVLink_command_ack_message(
        command=mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, result=mavlink.MAV_RESULT_ACCEPTED,
    ))
    _poll_until(link, 0.0, lambda: link._rate_acked)
    fake.received()
    link.poll(0.0)
    assert [m for m in fake.received() if m.get_type() == "COMMAND_LONG"] == []


def test_ekf2s_origin_puts_truth_in_the_estimates_frame(api, sim):
    """The datum trap from scripts/hover_error.py: without the rebase, truth and
    estimate differ by the offset between the two origins."""
    link, fake = api
    home = frames.GeodeticProjection(47.397742, 8.545594, 488.0)
    north, east = 3.0, -4.0
    lat, lon = home.reproject(north, east)
    fake.send(mavlink.MAVLink_gps_global_origin_message(
        latitude=int(round(lat * 1e7)), longitude=int(round(lon * 1e7)),
        altitude=int(round(489.5 * 1e3)),
    ))
    _poll_until(link, 0.0, lambda: link.origin is not None)

    samples: list[Sample] = []
    host = ControllerHost(Scripted(), Schedule(period=0.02), IMU_DT,
                          estimates=link, home=home, recorder=samples.append)
    host.before_step(sim)
    truth = samples[0].truth
    # EKF2's origin sits 3 m north, 4 m west and 1.5 m above the home.
    expected = truth.state.pos_ned + np.array([-north, -east, 1.5])
    assert truth.pos_ned_ekf == pytest.approx(expected, abs=0.02)


# --- in the loop ----------------------------------------------------------


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _loop_config(**overrides) -> Config:
    cfg = Config(stub_physics=True, sidechannel_enabled=False, speed_factor=2.0,
                 max_sim_time=0.4, brake_timeout_s=0.2, status_interval_s=1e9)
    cfg.hil_port_base = _free_tcp_port()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _run(cfg: Config, host: ControllerHost, sidechannel=None, physics=None) -> LockstepLoop:
    # The fake PX4 of test_loop.py, which replies to every HIL_SENSOR.
    from test_loop import FakePX4

    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    fake = FakePX4(cfg.hil_port, reply=True)
    loop = LockstepLoop(cfg, physics or StubPhysics(cfg), server, sidechannel, None, host)
    try:
        fake.start()
        loop.run()
    finally:
        fake.stop()
        server.close()
    return loop


def test_the_loop_calls_the_controller_on_schedule_and_measures_px4s_lag():
    cfg = _loop_config()
    controller = Scripted(fn=lambda o: None)
    host = ControllerHost(controller, Schedule(period=0.02), cfg.imu_dt)
    loop = _run(cfg, host)
    assert len(controller.seen) == pytest.approx(loop.stats.frames / 5, abs=1)
    lags = loop.stats.px4_lag
    assert sum(lags.values()) > 0 and min(lags) >= 0
    assert "px4_lag=" in loop.stats.summary()


def test_controller_time_is_not_made_up_by_a_catch_up_burst():
    """A 10 ms controller every 5 frames at 2x: with the time handed back to the
    pacer, the run takes its simulated time at speed *plus* the compute. Without
    it, the pacer would race through the following frames to recover it."""
    cfg = _loop_config()

    def slow(obs):
        time.sleep(0.01)
        return None

    host = ControllerHost(Scripted(fn=slow), Schedule(period=0.02), cfg.imu_dt)
    loop = _run(cfg, host)
    assert loop.stats.wall_time >= loop.stats.sim_time / cfg.speed_factor + 0.8 * host.compute_total


class _OneArmCmd:
    """A side channel that delivers one arm_cmd and then nothing."""

    def __init__(self) -> None:
        from mujoco_px4_sitl.arm import ArmCommand
        self.pending = [ArmCommand(seq=99, values=np.zeros(2))]

    def poll(self):
        pending, self.pending = self.pending, []
        return pending

    def publish(self, *args) -> None:
        pass


class _Spy(StubPhysics):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.submitted = []

    def submit_arm_command(self, command) -> None:
        self.submitted.append(command.seq)


def test_side_channel_arm_cmd_does_not_compete_with_an_in_process_controller(caplog):
    cfg = _loop_config(max_sim_time=0.1)
    physics = _Spy(cfg)
    host = ControllerHost(Scripted(fn=lambda o: None), Schedule(period=0.02), cfg.imu_dt)
    with caplog.at_level(logging.WARNING):
        _run(cfg, host, sidechannel=_OneArmCmd(), physics=physics)
    assert 99 not in physics.submitted
    assert "arm_cmd seq 99 refused" in caplog.text


@pytest.mark.parametrize("kwargs", [
    {"schedule": Schedule(period=0.02)},
    {"recorder": print},
    {"controller": Scripted()},
])
def test_run_refuses_half_a_controller(kwargs):
    with pytest.raises(ValueError):
        run(Config(stub_physics=True), **kwargs)
