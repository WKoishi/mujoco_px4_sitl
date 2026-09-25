"""The arm writer, its command watchdog, and the propeller clearance check.

MuJoCo only, no PX4 build and no private geometry: every test builds a small
floating base with four disc-shaped rotor sites and a two-joint arm whose one
capsule can be swung into a disc.

What matters most here is behaviour that would otherwise be silent: a command
that drives nothing, a crashed controller that leaves the arm holding its last
command forever, a stale command accepted as fresh, and an arm passing through
a propeller disc without anyone noticing.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import time

import mujoco
import numpy as np
import pytest

from mujoco_px4_sitl.arm import (
    ArmCommand,
    ArmServos,
    _point_disc_distance,
    least_clearance,
    segment_disc_distance,
)
from mujoco_px4_sitl.config import Config
from mujoco_px4_sitl.sidechannel import SCHEMA_VERSION, SideChannel
from mujoco_px4_sitl.sim import MujocoPhysics, SimState

# Base at the origin of its own frame; discs of radius 0.1 in the plane z = 0 at
# (+-0.2, +-0.2). The arm hangs 0.05 below: arm_joint0 yaws about +z, arm_joint1
# pitches about +y (positive pitch moves the tip DOWN), and the capsule runs
# 0.25 along the link's +x with radius 0.02. The base's box does not collide,
# so the arm reaches anything its joints allow.
#
# At yaw 45 deg the arm points at rotor2. Its axis crosses the disc plane inside
# the disc for pitch in [-0.267, -0.201] rad; INTO_DISC is inside that window
# with room for the servo's gravity droop.
DISC_R = 0.1
CAPSULE_R = 0.02
ARM_DROP = 0.05
ARM_LENGTH = 0.25
INTO_DISC = np.array([math.pi / 4, -0.23])
CLEAR = np.array([math.pi / 4, 0.8])

_ROTORS = [(+0.2, -0.2), (-0.2, +0.2), (+0.2, +0.2), (-0.2, -0.2)]
_DISCS = "\n".join(
    f'      <site name="rotor{i}" type="cylinder" size="{DISC_R} 0.002" pos="{x} {y} 0"/>'
    for i, (x, y) in enumerate(_ROTORS)
)
_POINTS = "\n".join(
    f'      <site name="rotor{i}" pos="{x} {y} 0"/>' for i, (x, y) in enumerate(_ROTORS)
)
_ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="base_link" pos="0 0 1">
      <freejoint/>
      <geom name="core" type="box" size="0.1 0.1 0.02" mass="2.0" contype="0"
            conaffinity="0"/>
      <site name="imu"/>
{rotors}
      <body name="arm_link0" pos="0 0 -{drop}">
        <joint name="arm_joint0" axis="0 0 1" range="-3 3" damping="0.5" armature="0.01"/>
        <geom name="arm_link0_visual" type="sphere" size="0.01" mass="0.05"
              contype="0" conaffinity="0"/>
        <body name="arm_link1">
          <joint name="arm_joint1" axis="0 1 0" range="-1.5 1.5" damping="0.3"
                 armature="0.005"/>
          <geom name="arm_link1_col" type="capsule" fromto="0 0 0 {length} 0 0"
                size="{radius}" mass="0.2"/>
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


def arm_xml(rotors: str = _DISCS) -> str:
    return _ARM_XML.format(
        rotors=rotors, drop=ARM_DROP, length=ARM_LENGTH, radius=CAPSULE_R
    )


def physics(tmp_path, xml: str | None = None, **cfg) -> MujocoPhysics:
    """Base welded to the world, so only the arm moves -- and under gravity.

    Not --hold-pose: that re-pins the base only between frames, so within one
    the whole vehicle free-falls and the arm hangs weightless from it.
    """
    path = tmp_path / "arm.xml"
    path.write_text(xml or arm_xml())
    return MujocoPhysics(Config(model_path=path, **cfg))


@pytest.fixture
def sim(tmp_path) -> MujocoPhysics:
    return physics(tmp_path)


def joint_angles(sim: MujocoPhysics) -> np.ndarray:
    joints = sim.model.actuator_trnid[sim.arm.ids, 0]
    return np.array([sim.data.qpos[sim.model.jnt_qposadr[j]] for j in joints])


def run(sim: MujocoPhysics, seconds: float, stream: np.ndarray | None = None) -> None:
    """Step, optionally streaming ``stream`` at 50 Hz with state_time echoed."""
    for frame in range(int(round(seconds * 250))):
        if stream is not None and frame % 5 == 0:
            sim.submit_arm_command(
                ArmCommand(seq=frame, values=stream, state_time=sim.time)
            )
        sim.step_frame(np.zeros(4))


# --- the writer -----------------------------------------------------------


def test_arm_cmd_reaches_the_joints(sim: MujocoPhysics):
    """The gap this module closes: a command used to be parsed and dropped."""
    target = np.array([0.6, -0.4])
    run(sim, 1.5, stream=target)
    assert sim.arm.count == 2
    assert joint_angles(sim) == pytest.approx(target, abs=0.02)
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx(target)


def test_before_any_command_the_arm_holds_its_initial_pose(sim: MujocoPhysics):
    """Not "servo to zero": the targets are the pose the model starts in, which
    is zero only because this model's qpos0 is."""
    start = joint_angles(sim)
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx(start)
    run(sim, 0.5)
    assert sim.arm.command is None and not sim.arm.stale
    assert sim.arm_status().cmd_age is None


def test_a_command_drives_the_first_frame_after_it_arrives(sim: MujocoPhysics):
    sim.submit_arm_command(ArmCommand(values=np.array([0.3, 0.2])))
    sim.step_frame(np.zeros(4))
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx([0.3, 0.2])


@pytest.mark.parametrize("command, reason", [
    (ArmCommand(values=np.array([0.1])), "1 values for 2"),
    (ArmCommand(values=np.array([0.1, 0.2, 0.3])), "3 values for 2"),
    (ArmCommand(mode="torque", values=np.array([0.1, 0.2])), "position servos only"),
    (ArmCommand(values=np.array([0.1, math.nan])), "non-finite"),
    (ArmCommand(values=np.array([0.1, 0.2]), state_time=5.0), "echo ground_truth.time"),
    (ArmCommand(values=np.array([0.1, 0.2]), state_time=math.inf), "non-finite"),
])
def test_a_command_the_servos_cannot_honour_is_rejected_and_said_so(
    sim: MujocoPhysics, caplog, command, reason
):
    before = sim.data.ctrl[sim.arm.ids].copy()
    with caplog.at_level(logging.WARNING):
        assert not sim.arm.submit(command, sim.time)
    sim.step_frame(np.zeros(4))
    assert sim.arm.rejected == 1 and sim.arm.accepted == 0
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx(before)
    assert reason in caplog.text


def test_an_out_of_range_command_is_clamped_and_counted(sim: MujocoPhysics, caplog):
    """The servo cannot pass its range any more than ctrlrange lets it, but the
    controller should learn it asked."""
    with caplog.at_level(logging.WARNING):
        assert sim.arm.submit(ArmCommand(values=np.array([0.0, 2.5])), sim.time)
    sim.step_frame(np.zeros(4))
    assert sim.arm.clamped == 1
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx([0.0, 1.5])
    assert "clamped" in caplog.text


def test_an_older_state_time_than_the_command_in_force_is_out_of_order(sim: MujocoPhysics):
    run(sim, 0.1)
    assert sim.arm.submit(ArmCommand(values=np.array([0.2, 0.0]), state_time=0.08), sim.time)
    assert not sim.arm.submit(ArmCommand(values=np.array([0.9, 0.0]), state_time=0.04), sim.time)
    # Equal is fine: a controller may send several commands per state sample.
    assert sim.arm.submit(ArmCommand(values=np.array([0.3, 0.0]), state_time=0.08), sim.time)
    sim.step_frame(np.zeros(4))
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx([0.3, 0.0])


def test_a_model_with_no_arm_rejects_arm_cmd(tmp_path):
    xml = arm_xml().split("<actuator>")[0] + arm_xml().split("</actuator>")[1]
    sim = physics(tmp_path, xml)
    assert sim.arm.count == 0
    assert not sim.arm.submit(ArmCommand(values=np.array([0.1])), sim.time)


def test_an_arm_actuator_that_is_not_a_position_servo_is_refused(tmp_path):
    """arm_cmd values are angles; a motor would take one as a torque, silently."""
    xml = arm_xml().replace(
        '<position name="arm_act0" joint="arm_joint0" kp="20" kv="1" ctrlrange="-3 3"/>',
        '<motor name="arm_act0" joint="arm_joint0" ctrlrange="-3 3"/>',
    )
    with pytest.raises(ValueError, match="not a unit-gear position servo"):
        physics(tmp_path, xml)


def test_a_geared_servo_is_refused_since_it_would_scale_every_angle(tmp_path):
    xml = arm_xml().replace(
        '<position name="arm_act0" joint="arm_joint0" kp="20" kv="1" ctrlrange="-3 3"/>',
        '<position name="arm_act0" joint="arm_joint0" kp="20" kv="1" ctrlrange="-3 3" gear="2"/>',
    )
    with pytest.raises(ValueError, match="unit-gear position servo"):
        physics(tmp_path, xml)


def test_an_actuator_outside_the_contiguous_scan_is_called_out(tmp_path, caplog):
    xml = arm_xml().replace('name="arm_act1"', 'name="arm_act2"')
    with caplog.at_level(logging.WARNING):
        sim = physics(tmp_path, xml)
    assert sim.arm.count == 1
    assert "never written" in caplog.text


# --- the watchdog ---------------------------------------------------------


def test_freeze_retargets_to_the_reached_pose_once_commands_stop(tmp_path, caplog):
    sim = physics(tmp_path, arm_timeout_s=0.2, arm_on_timeout="freeze")
    run(sim, 1.0, stream=np.array([0.6, -0.4]))
    # A new, distant target, then silence: the watchdog must catch the arm
    # mid-swing rather than let it complete a move nobody is supervising.
    sim.submit_arm_command(ArmCommand(seq=999, values=np.array([-0.6, 0.4]), state_time=sim.time))
    with caplog.at_level(logging.WARNING):
        run(sim, 0.21)
    assert sim.arm.stale and sim.arm.stale_episodes == 1
    frozen = sim.data.ctrl[sim.arm.ids].copy()
    assert frozen[0] > -0.55, "froze at the target, not where the arm was"
    assert frozen == pytest.approx(np.array(sim.data.actuator_length[sim.arm.ids]), abs=0.05)
    run(sim, 1.0)
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx(frozen)
    assert "arm_cmd stale" in caplog.text


def test_keep_holds_the_last_target_but_still_reports_staleness(tmp_path):
    """The no-watchdog servo bus. Reporting is what distinguishes it from the
    old behaviour, where nothing said the controller had gone."""
    sim = physics(tmp_path, arm_timeout_s=0.2, arm_on_timeout="keep")
    run(sim, 0.5, stream=np.array([0.6, -0.4]))
    run(sim, 0.5)
    assert sim.arm.stale
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx([0.6, -0.4])
    assert sim.arm_status().cmd_stale


def test_limp_drops_servo_torque_and_resuming_restores_it(tmp_path):
    sim = physics(tmp_path, arm_timeout_s=0.2, arm_on_timeout="limp")
    gains = np.array(sim.model.actuator_gainprm[sim.arm.ids, 0])
    run(sim, 1.0, stream=np.array([0.0, -1.0]))  # arm raised against gravity
    raised = joint_angles(sim)[1]
    run(sim, 1.0)
    assert sim.arm.stale
    assert np.all(sim.model.actuator_gainprm[sim.arm.ids] == 0.0)
    assert np.all(sim.data.actuator_force[sim.arm.ids] == 0.0)
    assert joint_angles(sim)[1] > raised + 0.3, "a limp arm falls"
    run(sim, 1.5, stream=np.array([0.0, -1.0]))
    assert not sim.arm.stale
    assert sim.model.actuator_gainprm[sim.arm.ids, 0] == pytest.approx(gains)
    assert joint_angles(sim)[1] == pytest.approx(-1.0, abs=0.05)


def test_the_next_fresh_command_resumes_control(tmp_path, caplog):
    sim = physics(tmp_path, arm_timeout_s=0.2)
    run(sim, 0.3, stream=np.array([0.2, 0.0]))
    run(sim, 0.5)
    assert sim.arm.stale
    with caplog.at_level(logging.WARNING):
        run(sim, 0.1, stream=np.array([0.5, 0.1]))
    assert not sim.arm.stale and sim.arm.stale_episodes == 1
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx([0.5, 0.1])
    assert "resumed" in caplog.text


def test_a_command_computed_from_old_state_is_stale_on_arrival(tmp_path):
    """Age comes from state_time when the sender gives one, so a controller that
    keeps sending but stopped reading state is caught too."""
    sim = physics(tmp_path, arm_timeout_s=0.2)
    run(sim, 1.0)
    assert sim.arm.submit(ArmCommand(values=np.array([0.7, 0.0]), state_time=0.5), sim.time)
    sim.step_frame(np.zeros(4))
    assert sim.arm.stale
    assert sim.data.ctrl[sim.arm.ids] != pytest.approx([0.7, 0.0])
    assert sim.arm_status().cmd_age == pytest.approx(sim.time - 0.5)


def test_without_state_time_age_runs_from_arrival(tmp_path):
    sim = physics(tmp_path, arm_timeout_s=0.2)
    run(sim, 1.0)
    sim.submit_arm_command(ArmCommand(values=np.array([0.1, 0.0])))
    arrived = sim.time
    run(sim, 0.1)
    assert sim.arm_status().cmd_age == pytest.approx(sim.time - arrived)
    assert not sim.arm.stale


def test_timeout_zero_turns_the_watchdog_off(tmp_path):
    sim = physics(tmp_path, arm_timeout_s=0.0)
    run(sim, 0.2, stream=np.array([0.4, 0.0]))
    run(sim, 2.0)
    assert not sim.arm.stale and sim.arm.stale_episodes == 0
    assert sim.data.ctrl[sim.arm.ids] == pytest.approx([0.4, 0.0])


def test_the_watchdog_runs_on_simulated_time_not_wall_time(tmp_path):
    """A lockstep stall freezes the vehicle's clock; the arm driver on that
    vehicle must not see time pass while it is frozen."""
    sim = physics(tmp_path, arm_timeout_s=0.05)
    sim.submit_arm_command(ArmCommand(values=np.array([0.4, 0.0]), state_time=sim.time))
    sim.step_frame(np.zeros(4))
    time.sleep(0.1)
    sim.step_frame(np.zeros(4))
    assert not sim.arm.stale


def test_watchdog_settings_are_validated():
    with pytest.raises(ValueError, match="arm_on_timeout"):
        Config(stub_physics=True, arm_on_timeout="hold").validate()
    with pytest.raises(ValueError, match="arm_timeout_s"):
        Config(stub_physics=True, arm_timeout_s=-1.0).validate()
    model = mujoco.MjModel.from_xml_string(arm_xml())
    with pytest.raises(ValueError, match="not in"):
        ArmServos(model, 0.5, "hold")


# --- exact distance -------------------------------------------------------


def _dense_minimum(a, b, c, n, radius) -> float:
    """Brute force: dense sampling, then a golden-section polish of the best
    bracket (the function is convex along the segment)."""
    t = np.linspace(0.0, 1.0, 20001)
    f = _point_disc_distance(a + t[:, None] * (b - a), c, n, radius)
    k = int(np.argmin(f))
    lo, hi = t[max(k - 1, 0)], t[min(k + 1, t.size - 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def at(s: float) -> float:
        return float(_point_disc_distance(a + s * (b - a), c, n, radius))

    for _ in range(80):
        m1, m2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if at(m1) < at(m2):
            hi = m2
        else:
            lo = m1
    return min(float(f.min()), at(0.5 * (lo + hi)))


def _random_pairs(seed: int, count: int):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(count, 3)) * 0.1
    n = rng.normal(size=(count, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    radius = rng.uniform(0.05, 0.3, size=count)
    a = c + rng.uniform(-0.4, 0.4, size=(count, 3))
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    b = a + d * rng.uniform(0.0, 0.3, size=count)[:, None]
    return a, b, c, n, radius


def test_segment_disc_distance_is_exact_against_brute_force():
    a, b, c, n, radius = _random_pairs(0, 400)
    got = segment_disc_distance(a, b, c, n, radius)
    for i in range(len(a)):
        assert got[i] == pytest.approx(_dense_minimum(a[i], b[i], c[i], n[i], radius[i]), abs=1e-12)


@pytest.mark.parametrize("a, b, expected", [
    # Along the normal, straight through the centre: pierces.
    ([0.0, 0.0, -0.1], [0.0, 0.0, 0.1], 0.0),
    # Along the normal, beside the rim.
    ([0.3, 0.0, 0.05], [0.3, 0.0, 0.2], math.hypot(0.05, 0.1)),
    # Zero length -- a sphere's centre -- above the disc.
    ([0.1, 0.0, 0.07], [0.1, 0.0, 0.07], 0.07),
    # In the plane, outside: distance to the rim.
    ([0.5, -0.1, 0.0], [0.5, 0.1, 0.0], 0.3),
    # Parallel to the plane, above it, crossing over the rim.
    ([0.1, 0.0, 0.03], [0.4, 0.0, 0.03], 0.03),
    # Passing outside the rim, tilted: the rim-circle stationary point.
    ([0.25, -0.3, -0.05], [0.25, 0.3, 0.05], 0.05),
])
def test_segment_disc_distance_special_cases(a, b, expected):
    """Unit disc of radius 0.2 at the origin, normal +z."""
    got = segment_disc_distance(
        np.array([a], dtype=float), np.array([b], dtype=float),
        np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([0.2]),
    )
    assert got[0] == pytest.approx(expected, abs=1e-12)


def test_least_clearance_is_the_minimum_over_rows():
    """Only rows that could hold the minimum get the quartic; the answer must
    not depend on that shortcut."""
    a, b, c, n, radius = _random_pairs(1, 2000)
    thickness = np.random.default_rng(2).uniform(0.0, 0.03, size=len(a))
    exact = segment_disc_distance(a, b, c, n, radius) - thickness
    rng = np.random.default_rng(3)
    for _ in range(100):
        rows = rng.choice(len(a), 24, replace=False)
        value, row = least_clearance(a[rows], b[rows], c[rows], n[rows], radius[rows], thickness[rows])
        assert value == pytest.approx(exact[rows].min(), abs=1e-12)
        assert exact[rows][row] == pytest.approx(value, abs=1e-12)


# --- propeller clearance --------------------------------------------------


def pose(sim: MujocoPhysics, yaw: float, pitch: float) -> None:
    """Put the arm at a pose directly, bypassing the servos."""
    joints = sim.model.actuator_trnid[sim.arm.ids, 0]
    for joint, angle in zip(joints, (yaw, pitch)):
        sim.data.qpos[sim.model.jnt_qposadr[joint]] = angle
    mujoco.mj_forward(sim.model, sim.data)


def test_clearance_matches_the_geometry(sim: MujocoPhysics):
    """Swung under rotor2 and pitched up until the tip is 10 mm below the disc
    plane: the tip is inside the disc radially, so the clearance is that gap
    minus the capsule radius."""
    gap = 0.010
    pitch = -math.asin((ARM_DROP - gap) / ARM_LENGTH)
    pose(sim, math.pi / 4, pitch)
    clearance, pair = sim.propellers.measure(sim.data)
    assert clearance == pytest.approx(gap - CAPSULE_R, abs=1e-9)
    assert sim.propellers.pair_names(pair) == ("arm_link1_col", "rotor2")


def test_clearance_bottoms_out_once_the_axis_crosses_the_disc(sim: MujocoPhysics):
    pose(sim, *INTO_DISC)
    clearance, _ = sim.propellers.measure(sim.data)
    assert clearance == pytest.approx(-CAPSULE_R, abs=1e-12)


def test_an_intrusion_is_reported_and_not_blocked(sim: MujocoPhysics, caplog):
    """Commanded into rotor2's disc, the arm gets there: nothing clamps it. The
    intrusion is counted, logged when it starts and when it ends."""
    run(sim, 0.5, stream=CLEAR)
    assert sim.propellers.clearance > 0.0
    with caplog.at_level(logging.WARNING):
        run(sim, 1.5, stream=INTO_DISC)
        assert joint_angles(sim) == pytest.approx(INTO_DISC, abs=0.03)
        assert sim.propellers.clearance < 0.0
        assert sim.propellers.intrusions == 1
        run(sim, 1.0, stream=CLEAR)
    assert sim.propellers.clearance > 0.0
    assert sim.propellers.intrusions == 1
    assert sim.propellers.deepest == pytest.approx(CAPSULE_R, abs=1e-6)
    assert "propeller intrusion #1 at" in caplog.text
    assert "propeller intrusion #1 ended" in caplog.text
    assert "rotor2" in caplog.text


def test_without_disc_sites_the_check_says_it_is_not_running(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        sim = physics(tmp_path, arm_xml(rotors=_POINTS))
    assert not sim.propellers.enabled
    assert "NOT checked" in caplog.text
    assert sim.arm_status().prop_clearance is None


def test_a_model_without_an_arm_reports_no_arm_status():
    sim = MujocoPhysics(Config())  # the quad
    assert sim.arm_status() is None


def test_status_carries_the_numbers_the_side_channel_publishes(sim: MujocoPhysics):
    run(sim, 0.2, stream=np.array([0.1, 0.1]))
    status = sim.arm_status()
    assert status.cmd_seq is not None and status.cmd_age is not None
    assert status.prop_clearance == pytest.approx(sim.propellers.clearance)
    assert "prop_clearance=" in status.summary()


# --- the side channel -----------------------------------------------------


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def channel():
    server = SideChannel("127.0.0.1", _free_udp_port())
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.bind(("127.0.0.1", 0))
    client.settimeout(1.0)
    yield server, client
    client.close()
    server.close()


def _send(client: socket.socket, server: SideChannel, payload) -> None:
    blob = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    client.sendto(blob, server.sock.getsockname())


def _poll_until(server: SideChannel, count: int):
    got = []
    deadline = time.monotonic() + 1.0
    while len(got) < count and time.monotonic() < deadline:
        got += server.poll()
    return got


def test_arm_cmds_come_out_in_arrival_order_with_state_time(channel):
    server, client = channel
    for seq, stamp in ((1, 0.2), (2, 0.1)):
        _send(client, server, {"v": SCHEMA_VERSION, "type": "arm_cmd", "seq": seq,
                               "values": [0.1, 0.2], "state_time": stamp})
    got = _poll_until(server, 2)
    assert [c.seq for c in got] == [1, 2]
    assert [c.state_time for c in got] == [0.2, 0.1]
    assert got[0].values == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("payload", [
    {"values": ["a", "b"]},
    {"values": "0.1,0.2"},
    {"values": [[0.1], [0.2]]},
    {"values": [0.1], "seq": "one"},
    {"values": [0.1], "state_time": "now"},
    {"values": [0.1], "mode": 3},
])
def test_a_malformed_arm_cmd_is_dropped_without_taking_the_loop_down(channel, caplog, payload):
    """This runs inside lockstep: an exception here stops PX4's clock too."""
    server, client = channel
    with caplog.at_level(logging.WARNING):
        _send(client, server, {"v": SCHEMA_VERSION, "type": "arm_cmd", **payload})
        _send(client, server, {"v": SCHEMA_VERSION, "type": "arm_cmd", "values": [0.3]})
        got = _poll_until(server, 1)
    assert [list(c.values) for c in got] == [[0.3]]
    assert "malformed arm_cmd" in caplog.text


def test_ground_truth_carries_the_arm_block_only_when_there_is_an_arm(channel, sim):
    server, client = channel
    _send(client, server, {"v": SCHEMA_VERSION, "type": "subscribe"})
    _poll_until(server, 0)
    deadline = time.monotonic() + 1.0
    while not server.subscribers and time.monotonic() < deadline:
        server.poll()

    run(sim, 0.1, stream=np.array([0.1, 0.1]))
    server.publish(sim.state(), np.zeros(4), sim.arm_status())
    with_arm = json.loads(client.recvfrom(65507)[0])
    server.publish(SimState(), np.zeros(4), None)
    without = json.loads(client.recvfrom(65507)[0])

    assert set(with_arm["arm"]) == {"cmd_seq", "cmd_age", "cmd_stale", "prop_clearance"}
    assert with_arm["arm"]["cmd_stale"] is False
    assert with_arm["arm"]["prop_clearance"] == pytest.approx(sim.propellers.clearance, abs=1e-6)
    assert "arm" not in without
