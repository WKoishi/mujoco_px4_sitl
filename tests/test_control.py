"""The in-process controller host, its schedule, and PX4's API link.

MuJoCo only for the host (a welded base and a two-joint arm, as in
``test_arm.py``), fake PX4 peers over UDP for the API link, and the stub physics
for the loop. No PX4 build, no private geometry.

What matters most here is timing that would otherwise be silent: a command,
setpoint or estimate a frame early or late, a drop schedule that loses more in a
row than it promised, an age bound that the arm's own watchdog does not honour,
an out-of-order estimate taken for the newest, and a barrier that returns
before PX4 has echoed it.
"""

from __future__ import annotations

import logging
import math
import socket
import threading
import time

import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mavlink

from mujoco_px4_sitl import frames, px4link
from mujoco_px4_sitl.config import Config
from mujoco_px4_sitl.control import (
    CappedDrops,
    ControllerHost,
    ControllerOutput,
    Observation,
    Sample,
    Schedule,
)
from mujoco_px4_sitl.loop import LockstepLoop
from mujoco_px4_sitl.main import run
from mujoco_px4_sitl.px4link import ApiLink, CommandAck, Px4Estimate
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
    {"period": 0.02, "estimate_delay": 0.0}, {"period": 0.02, "setpoint_delay": 0.0},
    {"period": 0.02, "estimate_delay": 0.006},
])
def test_a_schedule_off_the_imu_grid_is_refused(kwargs):
    with pytest.raises(ValueError, match="IMU frames"):
        Schedule(**kwargs).frames(IMU_DT)


def test_px4_delays_default_to_one_frame_the_least_allowed():
    frames = Schedule(period=0.02).frames(IMU_DT)
    assert (frames.estimate_delay, frames.setpoint_delay) == (1, 1)
    frames = Schedule(period=0.02, estimate_delay=0.012, setpoint_delay=0.008).frames(IMU_DT)
    assert (frames.estimate_delay, frames.setpoint_delay) == (3, 2)


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


# --- PX4's legs, against a stand-in link ---------------------------------


def an_estimate(t: float, x: float = 0.0) -> Px4Estimate:
    return Px4Estimate(
        time=t, received=t, pos_ned=np.array([x, 0.0, 0.0]), vel_ned=np.zeros(3),
        q_frd_ned=np.array([1.0, 0.0, 0.0, 0.0]), rates_frd=np.zeros(3),
        reset_counter=0, quality=100,
    )


class StubLink:
    """:class:`ApiLink`'s interface without sockets: estimates arrive when the
    test says, and every send is recorded with the frame it went out before."""

    def __init__(self) -> None:
        self._estimates: list[Px4Estimate] = []
        self.latest = None
        self.received = 0
        self.acks: list[CommandAck] = []
        self.origin = None
        self.sent: list[tuple[float, list]] = []
        self.ready = True

    def arrive(self, t: float, x: float = 0.0) -> None:
        self.latest = an_estimate(t, x)
        self._estimates.append(self.latest)
        self.received += 1

    def poll(self, now: float):
        return self.latest

    def fetch(self, now: float) -> bool:
        return self.ready

    def estimate_at(self, t: float):
        older = [e for e in self._estimates if e.time <= t + 1e-9]
        return older[-1] if older else None

    def send(self, messages, now: float) -> bool:
        self.sent.append((now, list(messages)))
        return self.ready

    def summary(self) -> str:
        return "api -"


def test_the_estimate_is_the_newest_at_or_before_the_chosen_delay(sim):
    """EKF2 every other frame: aged the delay or one frame more, never less."""
    link = StubLink()
    controller = Scripted(fn=lambda o: None)
    host = ControllerHost(controller, Schedule(period=IMU_DT, estimate_delay=2 * IMU_DT),
                          IMU_DT, link=link)
    for frame in range(12):
        if frame % 2 == 0:
            link.arrive(frame * IMU_DT)  # this frame's own output, already in
        host.before_step(sim, answered=True)
        sim.step_frame(np.zeros(4))
    ages = [round(o.estimate_age / IMU_DT) for o in controller.seen if o.estimate is not None]
    assert controller.seen[0].estimate is None and controller.seen[1].estimate is None
    assert ages == [2, 3] * 5
    assert host.over_age == 0 and host.estimate_age_bound == pytest.approx(3 * IMU_DT)
    # Proven once d + 2 frames have fetched: every output it could need is in.
    assert [o.proven for o in controller.seen] == [False] * 3 + [True] * 9


def test_an_estimate_that_arrives_after_its_sample_marks_the_sample_late(sim):
    """Completeness is checked after the fact: the record says so, and the
    sample reaches the recorder only once that is decided."""
    link = StubLink()
    samples: list[Sample] = []
    host = ControllerHost(Scripted(fn=lambda o: None), Schedule(period=IMU_DT), IMU_DT,
                          link=link, recorder=samples.append)
    link.arrive(0.0)
    host.before_step(sim, answered=True)  # t=0: cutoff -4 ms, nothing qualifies
    sim.step_frame(np.zeros(4))
    host.before_step(sim, answered=True)  # t=4: cutoff 0, the output of 0 is in
    sim.step_frame(np.zeros(4))
    assert [s.obs.index for s in samples] == [0]  # sample 1 is not decided yet
    host.before_step(sim, answered=True)  # t=8: cutoff 4, but 4's output is late
    sim.step_frame(np.zeros(4))
    link.arrive(0.004)
    host.before_step(sim, answered=True)  # t=12: it arrived after sample 2
    sim.step_frame(np.zeros(4))
    host.close()
    assert [(s.obs.index, s.estimate_late) for s in samples] == [
        (0, False), (1, False), (2, True), (3, False),
    ]
    assert samples[2].obs.estimate.time == 0.0
    assert samples[2].obs.proven
    assert host.late == 1


def test_a_sample_is_proven_only_after_d_plus_2_fetched_frames(sim):
    """An unanswered frame or an unechoed fetch restarts the count: the outputs
    it would have fetched are gone, since a fetch brings only the newest."""
    link = StubLink()
    controller = Scripted(fn=lambda o: None)
    host = ControllerHost(controller, Schedule(period=IMU_DT), IMU_DT, link=link)
    pattern = [(True, True)] * 4 + [(False, True)] + [(True, True)] * 3 + \
              [(True, False)] + [(True, True)] * 4
    for answered, echoed in pattern:
        link.ready = echoed
        host.before_step(sim, answered=answered)
    assert [o.proven for o in controller.seen] == (
        [False, False, True, True] + [False] + [False, False, True] + [False]
        + [False, False, True, True]
    )
    assert host.unproven == 8


@pytest.mark.parametrize("delay_frames", [1, 2, 5])
def test_setpoints_go_in_before_the_frame_the_delay_names(sim, delay_frames):
    """What the controller returns at t_k is sent before the HIL_SENSOR of
    t_k + setpoint_delay -- the loop's before_sensor of that frame."""
    link = StubLink()

    def rate(obs):
        return ControllerOutput(px4=[px4link.body_rate_setpoint([0, 0, 0], obs.index * 0.01)])

    host = ControllerHost(Scripted(fn=rate), Schedule(period=2 * IMU_DT,
                          setpoint_delay=delay_frames * IMU_DT), IMU_DT, link=link)
    for _ in range(20):
        host.before_sensor(sim.time)
        host.before_step(sim, answered=True)
        sim.step_frame(np.zeros(4))
    sent = [(round(now / IMU_DT), round(msgs[0].thrust / 0.01)) for now, msgs in link.sent]
    assert sent == [(2 * k + delay_frames, k) for k in range(len(sent))]
    assert len(sent) >= 7 and host.px4_unproven == 0


def test_arm_targets_and_px4_messages_travel_separately(sim):
    link = StubLink()
    samples: list[Sample] = []
    arm_cmd = px4link.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)
    host = ControllerHost(
        Scripted(fn=lambda o: ControllerOutput(arm=[0.1, 0.2], px4=[arm_cmd])),
        Schedule(period=IMU_DT, delay=2 * IMU_DT), IMU_DT, link=link, recorder=samples.append,
    )
    host.before_step(sim, answered=True)
    host.close()
    assert samples[0].command == pytest.approx([0.1, 0.2])
    assert samples[0].apply_time == pytest.approx(2 * IMU_DT)
    assert samples[0].px4 == (arm_cmd,) and samples[0].px4_time == pytest.approx(IMU_DT)


@pytest.mark.parametrize("msg", [
    px4link.command(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 31, 1e4),
    px4link.command(mavlink.MAV_CMD_REQUEST_MESSAGE, mavlink.MAVLINK_MSG_ID_HOME_POSITION),
    px4link.command(mavlink.MAV_CMD_GET_HOME_POSITION),
    mavlink.MAVLink_set_gps_global_origin_message(0, 473977420, 85455940, 488000),
])
def test_requests_that_block_px4s_receive_thread_are_refused(sim, msg):
    """They end in configure_stream_threadsafe(), which sleeps on simulated
    time: a barrier behind one never echoes while the clock is held."""
    host = ControllerHost(Scripted(fn=lambda o: ControllerOutput(px4=[msg])),
                          Schedule(period=IMU_DT), IMU_DT, link=StubLink())
    with pytest.raises(ValueError, match="configure_stream_threadsafe"):
        host.before_step(sim, answered=True)


def test_acks_reach_the_next_sample_once(sim):
    link = StubLink()
    controller = Scripted(fn=lambda o: None)
    host = ControllerHost(controller, Schedule(period=IMU_DT), IMU_DT, link=link)
    host.before_step(sim)
    link.acks.append(CommandAck(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0.004))
    host.before_step(sim)
    host.before_step(sim)
    assert [len(o.acks) for o in controller.seen] == [0, 1, 0]
    assert controller.seen[1].acks[0].command == mavlink.MAV_CMD_COMPONENT_ARM_DISARM


# --- the API link ------------------------------------------------------------


class FakeApiLink:
    """PX4's end of the API link, driven by the test: a UDP socket sending
    MAVLink 2 to ours."""

    def __init__(self, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(1.0)
        self.target = ("127.0.0.1", port)
        self.mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        self.mav.robust_parsing = True

    def send(self, msg, to=None) -> None:
        self.sock.sendto(msg.pack(self.mav), to or self.target)

    def heartbeat(self) -> None:
        self.send(mavlink.MAVLink_heartbeat_message(
            type=mavlink.MAV_TYPE_OCTOROTOR, autopilot=mavlink.MAV_AUTOPILOT_PX4,
            base_mode=0, custom_mode=0, system_status=mavlink.MAV_STATE_STANDBY,
            mavlink_version=3,
        ))

    def odometry(self, time_usec: int, x: float = 1.0, frame=mavlink.MAV_FRAME_LOCAL_NED,
                 child=mavlink.MAV_FRAME_LOCAL_NED, reset: int = 0) -> None:
        self.send(odometry_message(time_usec, x, frame, child, reset))

    def ack(self, command: int, result: int = mavlink.MAV_RESULT_ACCEPTED) -> None:
        self.send(mavlink.MAVLink_command_ack_message(command=command, result=result))

    def received(self, timeout: float = 0.2) -> list:
        out = []
        self.sock.settimeout(timeout)
        try:
            while True:
                data, _ = self.sock.recvfrom(65535)
                out.extend(self.mav.parse_buffer(data) or [])
        except (TimeoutError, socket.timeout):
            return out

    def close(self) -> None:
        self.sock.close()


def odometry_message(time_usec: int, x: float = 1.0, frame=mavlink.MAV_FRAME_LOCAL_NED,
                     child=mavlink.MAV_FRAME_LOCAL_NED, reset: int = 0):
    return mavlink.MAVLink_odometry_message(
        time_usec=time_usec, frame_id=frame, child_frame_id=child,
        x=x, y=2.0, z=-3.0, q=[1.0, 0.0, 0.0, 0.0], vx=0.1, vy=0.2, vz=0.3,
        rollspeed=0.01, pitchspeed=0.02, yawspeed=0.03,
        pose_covariance=[math.nan] * 21, velocity_covariance=[math.nan] * 21,
        reset_counter=reset, estimator_type=mavlink.MAV_ESTIMATOR_TYPE_AUTOPILOT,
        quality=100,
    )


class FakeApiPeer(threading.Thread):
    """PX4's receive thread, as plan 3.9 describes it: handles each datagram's
    messages in order, echoes PING from the same thread, acknowledges the
    silencing request, and answers ``REQUEST_MESSAGE(ODOMETRY)`` with
    ``odometry()``'s sample time, if it gives one.

    Every message is logged with ``clock()`` at its arrival: PX4's clock, when a
    fake PX4 shares it. While ``blocked`` is set it reads nothing, the way the
    real thread sleeps in ``configure_stream_threadsafe()``; the queued datagrams
    are handled once it clears.
    """

    def __init__(self, port: int, clock=lambda: -1, odometry=None) -> None:
        super().__init__(daemon=True)
        self.link = FakeApiLink(port)
        self.clock = clock
        self.odometry = odometry
        self.log: list[tuple[str, object, int]] = []
        self.blocked = threading.Event()
        self._queued: list = []
        self._halt = threading.Event()

    def run(self) -> None:
        last_heartbeat = 0.0
        self.link.sock.settimeout(0.01)
        while not self._halt.is_set():
            if time.monotonic() - last_heartbeat > 0.05:
                self.link.heartbeat()
                last_heartbeat = time.monotonic()
            if self.blocked.is_set():
                time.sleep(0.001)
                continue
            while self._queued:
                self._handle(*self._queued.pop(0))
            try:
                data, addr = self.link.sock.recvfrom(65535)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            for msg in self.link.mav.parse_buffer(data) or []:
                if self.blocked.is_set():
                    self._queued.append((msg, addr))
                else:
                    self._handle(msg, addr)

    def _handle(self, msg, addr) -> None:
        kind = msg.get_type()
        self.log.append((kind, msg, self.clock()))
        if kind == "PING" and (msg.target_system, msg.target_component) == (0, 0):
            self.link.send(mavlink.MAVLink_ping_message(
                msg.time_usec, msg.seq, msg.get_srcSystem(), msg.get_srcComponent()), addr)
        elif kind == "COMMAND_LONG" and msg.command == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            self.link.send(mavlink.MAVLink_command_ack_message(
                command=msg.command, result=mavlink.MAV_RESULT_ACCEPTED), addr)
        elif (kind == "COMMAND_LONG" and msg.command == mavlink.MAV_CMD_REQUEST_MESSAGE
              and self.odometry is not None):
            sample = self.odometry()
            if sample is not None:
                self.link.send(odometry_message(sample), addr)

    def of(self, kind: str) -> list:
        return [entry for entry in self.log if entry[0] == kind]

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)
        self.link.close()


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def api():
    port = _free_udp_port()
    link = ApiLink("127.0.0.1", port, barrier_timeout=0.1)
    fake = FakeApiLink(port)
    yield link, fake
    fake.close()
    link.close()


@pytest.fixture
def peer():
    port = _free_udp_port()
    link = ApiLink("127.0.0.1", port, barrier_timeout=0.1)
    fake = FakeApiPeer(port)
    fake.start()
    yield link, fake
    fake.stop()
    link.close()


def _poll_until(link: ApiLink, now: float, done, timeout: float = 2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        link.poll(now)
        if done():
            return
        time.sleep(0.005)
    raise AssertionError("API link never got there")


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
    assert link.estimate_at(0.0025).pos_ned[0] == 1.0
    assert link.estimate_at(0.003).pos_ned[0] == 3.0
    assert link.estimate_at(0.0009) is None


def test_odometry_not_in_local_ned_is_refused_not_rotated(api):
    link, fake = api
    fake.odometry(1000, child=mavlink.MAV_FRAME_BODY_FRD)
    _poll_until(link, 0.0, lambda: link.rejected == 1)
    assert link.latest is None


def test_the_stream_is_silenced_before_any_barrier(api, monkeypatch):
    """2e9 us, not -1, which would delete the stream; and no barrier until PX4
    has acknowledged, since the request blocks its receive thread till then."""
    link, fake = api
    monkeypatch.setattr(px4link, "_REQUEST_RETRY_S", 0.0)
    link.poll(0.0)
    assert fake.received(0.05) == []  # nowhere to send it before PX4 is heard

    fake.heartbeat()
    _poll_until(link, 0.0, lambda: link._target is not None)
    sent = fake.received()
    # Between two PINGs: the first registers us with PX4, whose ACK goes only
    # to components it has seen; the second's echo shows the request is done.
    assert [m.get_type() for m in sent] == ["PING", "COMMAND_LONG", "PING"], \
        "a second request while one is in flight"
    requests = [sent[1]]
    assert requests[0].command == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert requests[0].param1 == mavlink.MAVLINK_MSG_ID_ODOMETRY
    assert requests[0].param2 == pytest.approx(2e9)
    assert (requests[0].target_system, requests[0].target_component) == (1, 1)

    # Not ready: a send goes out unbarriered, with no PING to wait on.
    assert link.send([px4link.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)], 0.0) is False
    assert [m.get_type() for m in fake.received()] == ["COMMAND_LONG"]
    assert not link.ready and link.unbarriered == 1

    fake.ack(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL)
    time.sleep(0.05)
    link.poll(0.0)
    assert not link.ready, "ready before the request has left PX4's receive thread"
    fake.send(mavlink.MAVLink_ping_message(0, sent[2].seq, px4link.SOURCE_SYSTEM,
                                           px4link.SOURCE_COMPONENT))
    _poll_until(link, 0.0, lambda: link.ready)
    link.poll(0.0)
    assert [m for m in fake.received(0.05) if m.get_type() == "COMMAND_LONG"] == []


def test_a_refused_silencing_is_asked_again(api, monkeypatch):
    link, fake = api
    monkeypatch.setattr(px4link, "_REQUEST_RETRY_S", 0.0)
    fake.heartbeat()
    _poll_until(link, 0.0, lambda: link._target is not None)
    first = fake.received()
    fake.send(mavlink.MAVLink_ping_message(0, first[-1].seq, px4link.SOURCE_SYSTEM,
                                           px4link.SOURCE_COMPONENT))
    fake.ack(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, mavlink.MAV_RESULT_FAILED)
    again = []
    _poll_until(link, 0.0, lambda: again.extend(fake.received(0.02)) or again)
    assert [m.get_type() for m in again] == ["PING", "COMMAND_LONG", "PING"]
    assert not link.ready


def test_a_barrier_returns_only_once_px4_echoes_it(peer):
    link, fake = peer
    _poll_until(link, 0.0, lambda: link.ready)
    setpoint = px4link.body_rate_setpoint([0.1, 0.0, 0.0], 0.4)
    assert link.send([setpoint], 0.0) is True
    kinds = [k for k, _, _ in fake.log if k in ("SET_ATTITUDE_TARGET", "PING")]
    assert kinds[-2:] == ["SET_ATTITUDE_TARGET", "PING"]  # in order, in one handling
    msg = fake.of("SET_ATTITUDE_TARGET")[0][1]
    assert (msg.target_system, msg.target_component) == (1, 1)
    assert msg.body_roll_rate == pytest.approx(0.1) and msg.thrust == pytest.approx(0.4)
    assert (link.barriers, link.barrier_timeouts) == (1, 0)


def test_a_long_send_is_split_but_the_barrier_still_comes_last(peer):
    link, fake = peer
    _poll_until(link, 0.0, lambda: link.ready)
    many = [px4link.position_setpoint([k, 0.0, -3.0]) for k in range(40)]
    assert link.send(many, 0.0) is True
    kinds = [k for k, _, _ in fake.log if k in ("SET_POSITION_TARGET_LOCAL_NED", "PING")]
    assert kinds[-41:] == ["SET_POSITION_TARGET_LOCAL_NED"] * 40 + ["PING"]


def test_a_blocked_receive_thread_times_the_barrier_out_loudly(peer, caplog):
    """The stall the probes hit: PX4's receive thread asleep on simulated time.
    The barrier gives up after its wall-clock timeout, says so, and later sends
    do not wait again until an echo shows the thread is back."""
    link, fake = peer
    _poll_until(link, 0.0, lambda: link.ready)
    fake.blocked.set()
    time.sleep(0.03)  # past any recvfrom already under way
    with caplog.at_level(logging.WARNING):
        start = time.monotonic()
        assert link.send([px4link.body_rate_setpoint([0, 0, 0], 0.3)], 0.0) is False
        assert time.monotonic() - start >= 0.09
    assert "PING barrier not echoed" in caplog.text
    assert link.barrier_timeouts == 1

    start = time.monotonic()
    assert link.send([px4link.body_rate_setpoint([0, 0, 0], 0.3)], 0.0) is False
    assert time.monotonic() - start < 0.05, "waited again on a thread known to be stuck"
    assert link.unbarriered == 1

    fake.blocked.clear()  # the thread wakes, and handles its queue in order
    _poll_until(link, 0.0, lambda: link._barrier_ok)
    assert link.send([px4link.body_rate_setpoint([0, 0, 0], 0.3)], 0.0) is True
    assert link.barrier_timeouts == 1


def test_a_fetch_brings_the_newest_output_before_the_echo(peer):
    link, fake = peer
    fake.odometry = lambda: 8000
    _poll_until(link, 0.0, lambda: link.ready)
    assert link.fetch(0.008) is True
    assert link.latest is not None and link.latest.time == pytest.approx(0.008)
    request = fake.of("COMMAND_LONG")[-1][1]
    assert request.command == mavlink.MAV_CMD_REQUEST_MESSAGE
    assert request.param1 == mavlink.MAVLINK_MSG_ID_ODOMETRY


def test_a_fetch_before_the_link_is_ready_sends_nothing(api):
    link, fake = api
    assert link.fetch(0.0) is False
    assert fake.received(0.05) == []


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
                          link=link, home=home, recorder=samples.append)
    host.before_step(sim)
    host.close()
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


def _run(cfg: Config, host: ControllerHost, sidechannel=None, physics=None,
         fake=None) -> LockstepLoop:
    # The fake PX4 of test_loop.py, which replies to every HIL_SENSOR.
    from test_loop import FakePX4

    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    fake = fake or FakePX4(cfg.hil_port, reply=True)
    loop = LockstepLoop(cfg, physics or StubPhysics(cfg), server, sidechannel, None, host)
    try:
        fake.start()
        loop.run()
    finally:
        fake.stop()
        server.close()
    return loop


def test_the_loop_calls_the_controller_on_schedule_and_answers_every_frame():
    cfg = _loop_config()
    controller = Scripted(fn=lambda o: None)
    host = ControllerHost(controller, Schedule(period=0.02), cfg.imu_dt)
    loop = _run(cfg, host)
    assert len(controller.seen) == pytest.approx(loop.stats.frames / 5, abs=1)
    assert loop.stats.unproven == 0 and loop.stats.answered > 0
    assert "answered=" in loop.stats.summary()
    # Without an API link, a sample is proven by the answer alone.
    assert controller.seen[-1].proven


def test_px4s_legs_land_on_the_frames_the_schedule_names():
    """End to end, strict loop and real API link against a fake PX4 whose
    estimator publishes every frame's output before answering, as an idle PX4
    does. A setpoint returned at t_k must reach PX4's receive thread while its
    clock is still at t_k + d_sp - 1, i.e. before the HIL_SENSOR of t_k + d_sp;
    an estimate must be exactly estimate_delay old; nothing may be late."""
    from test_loop import FakePX4

    cfg = _loop_config(max_sim_time=0.5, speed_factor=0.0)
    cfg.px4_api_port_base = _free_udp_port()
    fake_px4 = FakePX4(cfg.hil_port, reply=True)
    peer = FakeApiPeer(cfg.px4_api_port, clock=lambda: fake_px4.clock_us,
                       odometry=lambda: fake_px4.clock_us)
    d_sp, d_est = 3, 2

    def act(obs):
        if not obs.proven:
            return None
        return ControllerOutput(px4=[px4link.body_rate_setpoint([0, 0, 0], obs.index * 1e-3)])

    controller = Scripted(fn=act)
    samples: list[Sample] = []
    link = ApiLink("127.0.0.1", cfg.px4_api_port, barrier_timeout=0.2)
    host = ControllerHost(
        controller, Schedule(period=2 * IMU_DT, setpoint_delay=d_sp * IMU_DT,
                             estimate_delay=d_est * IMU_DT),
        cfg.imu_dt, link=link, recorder=samples.append,
    )
    peer.start()
    try:
        loop = _run(cfg, host, fake=fake_px4)
    finally:
        peer.stop()
        link.close()

    proven = [o for o in controller.seen if o.proven]
    assert len(proven) > 40
    # Only the first proven samples, before any fetched output is old enough.
    assert all(o.estimate is not None for o in proven[2:])
    assert all(round(o.estimate_age / IMU_DT) == d_est for o in proven[2:])
    arrivals = [(round(m.thrust / 1e-3), clock) for _, m, clock in peer.of("SET_ATTITUDE_TARGET")]
    assert len(arrivals) > 40
    for index, clock in arrivals:
        assert clock == round((2 * index + d_sp - 1) * IMU_DT * 1e6), (index, clock)
    assert host.late == 0 and host.px4_unproven == 0 and link.barrier_timeouts == 0
    assert loop.stats.unproven == 0
    assert len(samples) == len(controller.seen)
    # The actuators in the record are the answer to the previous frame.
    strict = [s for s in samples if s.obs.proven]
    assert all(
        s.truth.actuators.time_usec == round((s.obs.time - IMU_DT) * 1e6) for s in strict
    )


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
