#!/usr/bin/env python3
"""Fly a mission scheduled on simulated time, with the controller in process.

Starts PX4 in a throw-away rootfs, so every run boots from the same parameters
and dataman, then runs the simulator in this process through
``mujoco_px4_sitl.main.run`` with a controller that sends PX4 its setpoints and
commands -- arming included -- at chosen simulated times. Every sample is
recorded with ground truth to an ``.npz``, and a report is printed.

Profiles:

``legs``       Phase 2's acceptance (``AGENTS.md`` section 3), and the probe's
               mission (``IMPLEMENTATION_PLAN.md`` phase 7, "The PX4 legs"):
               Offboard hold at 3 m, armed at 20 s; a 3 s body-rate window at
               30 s with a host attitude loop on the estimate, and a 3 s attitude
               window at 38 s, each with a seeded +-0.015 thrust PRBS per frame.
               Controller every frame, both PX4 delays one frame. Reports every
               frame's IMU -> actuator delay, which frame's setpoint reached the
               rotors, late estimates, and speed.
``arm-sweep``  Phase 7's in-process flight: Offboard hold at 5 m, armed at 20 s,
               ``arm_joint0`` swept +-0.5 rad at 0.2 Hz from 40 s to 70 s, Land at
               75 s. Period 20 ms, arm delay 8 ms, ``CappedDrops(0.2, 2, seed=1)``.
``steps``      Phase 5's attitude steps for the X8: yaw 20 and 90 degrees through
               position setpoints, 15 s each, then roll and pitch +-5 degrees
               through attitude setpoints for 3 s each, from level, with thrust
               at hover plus an altitude PD on the estimate. Read from truth.
``acceptance`` ``legs`` idle and with every core busy, then one table of the
               acceptance checks, each marked PASS or FAIL. The exit status is 0
               only if all pass. This is the one to run after touching the loop.

    python scripts/fly_in_process.py acceptance                # quad, about 40 s
    python scripts/fly_in_process.py legs --out /tmp/quad.npz
    python scripts/fly_in_process.py legs --load 16 --out /tmp/quad_load.npz
    python scripts/fly_in_process.py legs --model x8.xml --rotors x8.yaml \\
        --airframe 22002 --hover 0.2162 --out /tmp/x8.npz
    python scripts/fly_in_process.py --compare /tmp/a.npz /tmp/b.npz

``--load N`` keeps N processes spinning for the whole run: "with every core
busy" is where the probes found the barrier and the estimate delay necessary.

Output is the report only, plus the simulator's warnings: an unproven frame or
a barrier timeout still shows. ``--verbose`` brings back the simulator's log.

PX4 cannot finish exiting once simulated time has stopped (plan section 7), so
this kills it after the run; the rootfs is discarded, and the newest ulog is
copied next to the ``.npz`` first. Needs a built ``px4_sitl_default`` with the
airframe installed (``scripts/install_px4_files.sh``).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from pymavlink.dialects.v20 import common as mavlink

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mujoco_px4_sitl import px4link  # noqa: E402
from mujoco_px4_sitl.config import config_from_args  # noqa: E402
from mujoco_px4_sitl.control import (  # noqa: E402
    CappedDrops,
    ControllerOutput,
    Observation,
    Sample,
    Schedule,
)
from mujoco_px4_sitl.main import configure_logging, run  # noqa: E402
from mujoco_px4_sitl.rotorconfig import load_rotors  # noqa: E402

PX4_DIR = Path(os.environ.get("PX4_DIR", REPO.parent / "PX4-Autopilot"))
PX4_BIN = PX4_DIR / "build" / "px4_sitl_default" / "bin" / "px4"
PX4_ETC = PX4_DIR / "build" / "px4_sitl_default" / "etc"

PRBS_AMP = 0.015
OFFBOARD = 6.0  # PX4 custom main mode
KIND_POSITION, KIND_RATE, KIND_ATTITUDE = 1, 2, 3


def euler(q) -> tuple[float, float, float]:
    """Roll, pitch, yaw of a [w, x, y, z] body FRD -> NED quaternion."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


# --- missions ------------------------------------------------------------------


class Mission:
    """Offboard at a height: setpoints from 18 s, Offboard at 19.5 s, armed at
    20 s. Subclasses fill in what happens once airborne."""

    t_setpoints, t_mode, t_arm = 18.0, 19.5, 20.0

    def __init__(self, imu_dt: float, height: float, hover: float) -> None:
        self.imu_dt = imu_dt
        self.height = height
        self.hover = hover
        self.mode_sent = self.arm_sent = False
        # Per sample: what kind of setpoint went to PX4, and its thrust.
        self.kind: dict[int, int] = {}
        self.thrust: dict[int, float] = {}

    def commands(self, t: float) -> list:
        msgs = []
        if not self.mode_sent and t >= self.t_mode:
            msgs.append(px4link.command(mavlink.MAV_CMD_DO_SET_MODE, 1.0, OFFBOARD))
            self.mode_sent = True
        if not self.arm_sent and t >= self.t_arm:
            msgs.append(px4link.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0))
            self.arm_sent = True
        return msgs

    def hold(self, obs: Observation, every: int = 5, yaw: float | None = None) -> list:
        if obs.index % every:
            return []
        self.kind[obs.index] = KIND_POSITION
        return [px4link.position_setpoint([0.0, 0.0, -self.height], yaw)]

    def altitude_thrust(self, obs: Observation, z_ref: float, roll: float, pitch: float) -> float:
        est = obs.estimate
        tilt = max(np.cos(roll) * np.cos(pitch), 0.7)
        return float(np.clip(
            (self.hover + 0.06 * (est.pos_ned[2] - z_ref) + 0.05 * est.vel_ned[2]) / tilt,
            0.08, 0.6))


class Legs(Mission):
    rate_window = (30.0, 33.0)
    attitude_window = (38.0, 41.0)
    end = 46.0

    def __init__(self, imu_dt: float, hover: float, seed: int) -> None:
        super().__init__(imu_dt, 3.0, hover)
        self.rng = np.random.default_rng(seed)
        self.window_yaw = self.z_ref = None
        self.base_thrust = hover

    def step(self, obs: Observation):
        t = obs.time
        if t < self.t_setpoints:
            return None
        in_rate = self.rate_window[0] <= t < self.rate_window[1]
        in_att = self.attitude_window[0] <= t < self.attitude_window[1]
        msgs = []
        if (in_rate or in_att) and obs.estimate is not None:
            roll, pitch, yaw = euler(obs.estimate.q_frd_ned)
            if self.window_yaw is None:
                self.window_yaw, self.z_ref = yaw, obs.estimate.pos_ned[2]
            if obs.index % 5 == 0:  # the host's altitude loop, every 20 ms
                self.base_thrust = self.altitude_thrust(obs, self.z_ref, roll, pitch)
            thrust = self.base_thrust + PRBS_AMP * (1 if self.rng.random() < 0.5 else -1)
            self.thrust[obs.index] = thrust
            if in_rate:
                self.kind[obs.index] = KIND_RATE
                dyaw = np.arctan2(np.sin(self.window_yaw - yaw), np.cos(self.window_yaw - yaw))
                msgs.append(px4link.body_rate_setpoint(
                    [-5.0 * roll, -5.0 * pitch, 2.0 * dyaw], thrust))
            else:
                self.kind[obs.index] = KIND_ATTITUDE
                msgs.append(px4link.attitude_setpoint(quat(0.0, 0.0, self.window_yaw), thrust))
        else:
            self.window_yaw = None
            msgs += self.hold(obs)
        msgs += self.commands(t)
        return ControllerOutput(px4=msgs) if msgs else None


class ArmSweep(Mission):
    sweep = (40.0, 70.0)
    t_land = 75.0
    end = 95.0

    def __init__(self, imu_dt: float, hover: float) -> None:
        super().__init__(imu_dt, 5.0, hover)
        self.land_sent = False

    def step(self, obs: Observation):
        t = obs.time
        arm = np.zeros(len(obs.joints.q))
        if self.sweep[0] <= t < self.sweep[1]:
            arm[0] = 0.5 * np.sin(2 * np.pi * 0.2 * (t - self.sweep[0]))
        msgs = []
        if self.t_setpoints <= t < self.t_land:
            msgs += self.hold(obs, every=1)
            msgs += self.commands(t)
        if not self.land_sent and t >= self.t_land:
            msgs.append(px4link.command(mavlink.MAV_CMD_NAV_LAND))
            self.land_sent = True
        return ControllerOutput(arm=arm, px4=msgs)


class Steps(Mission):
    # (start, yaw deg) through position setpoints, each held 15 s.
    yaw_steps = [(35.0, 20.0), (50.0, 0.0), (65.0, 90.0), (80.0, 0.0)]
    # (start, axis, deg) through attitude setpoints, 3 s each, from level.
    attitude_start, attitude_end = 95.0, 131.0
    attitude_steps = [(98.0, "roll", 5.0), (104.0, "roll", -5.0),
                      (110.0, "pitch", 5.0), (116.0, "pitch", -5.0)]
    t_land = 131.0
    end = 145.0

    def __init__(self, imu_dt: float, hover: float) -> None:
        super().__init__(imu_dt, 3.0, hover)
        self.z_ref = None
        self.base_thrust = hover
        self.land_sent = False

    def yaw_target(self, t: float) -> float:
        deg = 0.0
        for start, value in self.yaw_steps:
            if t >= start:
                deg = value
        return np.radians(deg)

    def attitude_target(self, t: float) -> tuple[float, float]:
        roll = pitch = 0.0
        for start, axis, deg in self.attitude_steps:
            if start <= t < start + 3.0:
                roll, pitch = (np.radians(deg), 0.0) if axis == "roll" else (0.0, np.radians(deg))
        return roll, pitch

    def step(self, obs: Observation):
        t = obs.time
        if t < self.t_setpoints:
            return None
        msgs = []
        if self.attitude_start <= t < self.attitude_end and obs.estimate is not None:
            roll, pitch, _ = euler(obs.estimate.q_frd_ned)
            if self.z_ref is None:
                self.z_ref = obs.estimate.pos_ned[2]
            if obs.index % 5 == 0:
                self.base_thrust = self.altitude_thrust(obs, self.z_ref, roll, pitch)
            target = self.attitude_target(t)
            self.kind[obs.index] = KIND_ATTITUDE
            msgs.append(px4link.attitude_setpoint(quat(*target, 0.0), self.base_thrust))
        elif t < self.t_land:
            msgs += self.hold(obs, yaw=self.yaw_target(t))
        msgs += self.commands(t)
        if not self.land_sent and t >= self.t_land:
            msgs.append(px4link.command(mavlink.MAV_CMD_NAV_LAND))
            self.land_sent = True
        return ControllerOutput(px4=msgs) if msgs else None


# --- recording -----------------------------------------------------------------


class Recorder:
    """Keeps every sample as arrays; ``wall`` is when it reached the recorder."""

    def __init__(self) -> None:
        self.rows: dict[str, list] = {k: [] for k in (
            "index", "t", "proven", "late", "est_time", "est_pos", "est_reset", "truth_pos",
            "truth_pos_ekf", "truth_q", "joints", "cmd_age", "command", "dropped",
            "act_time", "act_armed", "act", "wall",
        )}

    def __call__(self, s: Sample) -> None:
        r, est, truth = self.rows, s.obs.estimate, s.truth
        r["index"].append(s.obs.index)
        r["t"].append(s.obs.time)
        r["proven"].append(s.obs.proven)
        r["late"].append(s.estimate_late)
        r["est_time"].append(np.nan if est is None else est.time)
        r["est_pos"].append(np.full(3, np.nan) if est is None else est.pos_ned)
        r["est_reset"].append(-1 if est is None else est.reset_counter)
        r["truth_pos"].append(truth.state.pos_ned)
        r["truth_pos_ekf"].append(np.full(3, np.nan) if truth.pos_ned_ekf is None
                                  else truth.pos_ned_ekf)
        r["truth_q"].append(truth.state.q_px4)
        r["joints"].append(truth.joints.q)
        age = None if truth.arm is None else truth.arm.cmd_age
        r["cmd_age"].append(np.nan if age is None else age)
        n = len(truth.joints.q)
        r["command"].append(np.full(n, np.nan) if s.command is None else s.command)
        r["dropped"].append(s.dropped)
        act = truth.actuators
        r["act_time"].append(-1 if act is None else act.time_usec)
        r["act_armed"].append(False if act is None else act.armed)
        r["act"].append(np.zeros(16) if act is None else act.controls[:16])
        r["wall"].append(time.perf_counter())

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.rows.items()}


class SummaryCatcher(logging.Handler):
    """Keeps the loop's final status line, which carries the host's counters."""

    line = ""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("loop stopped:"):
            SummaryCatcher.line = msg


def spin() -> None:
    # Forked after an earlier run() installed its SIGTERM handler, which only
    # stops a loop; this process has none to stop, and must die on terminate().
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    while True:
        pass


# --- PX4 -----------------------------------------------------------------------


class Px4:
    """PX4 in a throw-away rootfs, killed after the run."""

    def __init__(self, airframe: str, instance: int, log_path: Path) -> None:
        if not PX4_BIN.exists():
            raise SystemExit(f"{PX4_BIN} not found; build px4_sitl_default first")
        self.dir = tempfile.mkdtemp(prefix="px4_inproc_")
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [str(PX4_BIN), "-d", "-i", str(instance), "-w", self.dir, str(PX4_ETC)],
            env=dict(os.environ, PX4_SYS_AUTOSTART=airframe),
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )

    def stop(self, ulog_to: Path | None) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        self.log.close()
        logs = sorted(Path(self.dir, "log").rglob("*.ulg"))
        if ulog_to is not None and logs:
            shutil.copy(logs[-1], ulog_to)
        shutil.rmtree(self.dir, ignore_errors=True)


def setup_logging(verbose: bool) -> None:
    """The simulator's log goes to the console at WARNING, or everything with
    ``verbose``; the loop's final status line is caught either way."""
    configure_logging("INFO")
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setLevel(logging.INFO if verbose else logging.WARNING)
    root.addHandler(SummaryCatcher())
    # The quad's placeholder-motor warning: known, and on every quad run.
    logging.getLogger("mujoco_px4_sitl.vehicle").setLevel(logging.ERROR)


def fly(args) -> Path | None:
    """One run; returns the .npz it wrote."""
    if subprocess.run(["pgrep", "-x", "px4"], capture_output=True).returncode == 0:
        print("a px4 process is already running; stop it first", file=sys.stderr)
        return None
    sim_args = ["--model", str(args.model), "--no-sidechannel", "-s", str(args.speed),
                "--instance", str(args.instance), "--status-interval", "10"]
    if args.rotors is not None:
        sim_args += ["--rotors", str(args.rotors)]
    cfg = config_from_args(sim_args)
    n_rotors = 4 if args.rotors is None else len(load_rotors(args.rotors).spin)
    dt = cfg.imu_dt

    if args.profile in ("legs", "acceptance"):
        mission = Legs(dt, args.hover, args.seed)
        schedule = Schedule(period=dt)
    elif args.profile == "arm-sweep":
        mission = ArmSweep(dt, args.hover)
        schedule = Schedule(period=0.02, delay=0.008, drops=CappedDrops(0.2, 2, seed=1))
    else:
        mission = Steps(dt, args.hover)
        schedule = Schedule(period=dt)
    cfg.max_sim_time = mission.end

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spinners = [multiprocessing.Process(target=spin, daemon=True) for _ in range(args.load)]
    for p in spinners:
        p.start()
    recorder = Recorder()
    px4 = Px4(args.airframe, args.instance, out.with_suffix(".px4.log"))
    wall0 = time.monotonic()
    try:
        run(cfg, controller=mission, schedule=schedule, recorder=recorder)
    finally:
        px4.stop(out.with_suffix(".ulg"))
        for p in spinners:
            p.kill()
            p.join()
    data = recorder.arrays()
    meta = dict(vars(args), model=str(args.model), rotors=str(args.rotors), out=str(out),
                imu_dt=dt, n_rotors=n_rotors, wall=time.monotonic() - wall0,
                summary=SummaryCatcher.line,
                kind={str(k): v for k, v in mission.kind.items()},
                thrust={str(k): v for k, v in mission.thrust.items()})
    np.savez_compressed(out, meta=json.dumps(meta), **data)
    print(f"saved {len(data['t'])} samples to {out}", flush=True)
    return out


# --- reports -------------------------------------------------------------------


def load(path) -> dict:
    d = dict(np.load(path, allow_pickle=False))
    d["meta"] = json.loads(str(d["meta"]))
    return d


def counters(summary: str) -> dict[str, int]:
    """The integer counters of the loop's final status line. It carries two
    ``unproven``: the loop's frames first, then the controller's samples."""
    out: dict[str, int] = {}
    for key, value in re.findall(r"(\w+)=(-?\d+)(?=\s|$)", summary):
        if key == "unproven" and key in out:
            key = "samples_unproven"
        out.setdefault(key, int(value))
    return out


def timing(d) -> dict | None:
    """What every profile's report shares: PX4's legs, measured per sample."""
    m, t, act_time = d["meta"], d["t"], d["act_time"]
    dt = m["imu_dt"]
    answered = np.flatnonzero(act_time >= 0)
    if len(answered) == 0:
        return None
    first = answered[0]
    # A sample's actuators are the answer that drives its frame: one frame old
    # when PX4 answered the previous frame in time. The frame the first answer
    # arrives in runs on it at whatever age it arrived, since nothing waited on
    # the fallback frame before it (plan 3.2); it is reported apart, and the
    # one-frame guarantee counts from the frame after it.
    lag = np.round((t - act_time * 1e-6) / dt).astype(int)[first:]
    out = dict(first_answer=int(act_time[first] // round(dt * 1e6)),
               arrival_lag=int(lag[0]), lags=Counter(lag[1:].tolist()),
               samples=len(lag) - 1, proven_from=None)
    proven = d["proven"].astype(bool)
    if proven.any():
        p0 = np.flatnonzero(proven)[0]
        after = proven[p0:]
        known = after & np.isfinite(d["est_time"][p0:])
        late = d["late"].astype(bool)
        span_t, span_w = t[-1] - t[p0], d["wall"][-1] - d["wall"][p0]
        out.update(
            proven_from=float(t[p0]), proven=int(after.sum()), after=len(after),
            ages=Counter(np.round((t - d["est_time"]) / dt)[p0:][known].astype(int).tolist()),
            late_proven=int((late & proven).sum()), late_unproven=int((late & ~proven).sum()),
            speed=span_t / span_w if m["period_frames"] == 1 and span_w > 0 else None,
        )
    return out


def report(path) -> None:
    d = load(path)
    m = d["meta"]
    t = d["t"]
    c = counters(m["summary"])
    print(f"\n### {path}: {m['profile']}, {Path(m['model']).name}, load {m['load']}")
    print("loop: " + " ".join(f"{k}={c[k]}" for k in (
        "frames", "answered", "unproven", "brake", "timeouts", "samples_unproven",
        "late", "over_age", "px4_unproven", "barrier_timeouts") if k in c))
    tm = timing(d)
    if tm is None:
        print("PX4 never answered")
        return
    print(f"PX4's first answer (frame {tm['first_answer']}) drove its arrival frame at lag "
          f"{tm['arrival_lag']}; IMU -> actuator one frame after it on {tm['lags'][1]} of "
          f"{tm['samples']} samples; lags {dict(tm['lags'])}")
    if tm["proven_from"] is not None:
        print(f"proven: {tm['proven']} of {tm['after']} samples from t={tm['proven_from']:.2f} s; "
              f"estimate age {dict(tm['ages'])} frames; late {tm['late_proven']} proven, "
              f"{tm['late_unproven']} unproven"
              + (f"; speed x{tm['speed']:.2f}" if tm["speed"] else ""))
    armed = np.flatnonzero(d["act_armed"])
    if len(armed):
        print(f"first armed sample {armed[0]} (t={t[armed[0]]:.3f} s)")
    print(f"max altitude {-d['truth_pos'][:, 2].min():.2f} m")
    if m["profile"] in ("legs", "acceptance"):
        for kind, label in ((KIND_RATE, "body-rate"), (KIND_ATTITUDE, "attitude")):
            fit = setpoint_lags(d, kind)
            if fit is not None:
                coef, worst, per_frame = fit
                best = int(np.argmin(worst))
                print(f"{label} window: thrust reaches the rotors at lag {per_frame} "
                      f"(samples; worst residual {worst[best]:.3f} flips at lag {best})")
    if m["profile"] == "steps":
        step_report(d)
    if m["profile"] == "arm-sweep":
        airborne = d["act_armed"].astype(bool) & (-d["truth_pos"][:, 2] > 0.5)
        err = (d["est_pos"] - d["truth_pos_ekf"])[airborne]
        horiz = np.linalg.norm(err[:, :2], axis=1)
        vert = np.abs(err[:, 2])
        age = (t - d["est_time"])[np.isfinite(d["est_time"])]
        print(f"samples {len(t)}, dropped {int(d['dropped'].sum())}, cmd_age max "
              f"{np.nanmax(d['cmd_age']) * 1e3:.0f} ms; estimate age median "
              f"{np.median(age) * 1e3:.0f} ms, max {age.max() * 1e3:.0f} ms")
        if len(err):
            print(f"airborne {airborne.sum() * 0.02:.0f} s: estimate vs truth horizontal "
                  f"{np.nanmedian(horiz):.3f} median / {np.nanmax(horiz):.3f} max m, vertical "
                  f"{np.nanmedian(vert):.3f} / {np.nanmax(vert):.3f} m")
        tilt = np.degrees([np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))
                           for q in d["truth_q"]])
        sweep = (t >= ArmSweep.sweep[0]) & (t < ArmSweep.sweep[1])
        print(f"tilt during the sweep, max {tilt[sweep].max():.2f} deg")


def setpoint_lags(d, kind):
    """Which sample's thrust reached the rotors. The sum of u^2 is linear in
    collective thrust at THR_MDL_FAC 1 and blind to torque, so regress it on the
    thrust sent at samples k, k-1, ..., then score each single lag by its worst
    residual over the window, in units of one PRBS flip."""
    kinds, thrust = d["meta"]["kind"], d["meta"]["thrust"]
    idx = np.array(sorted(int(k) for k, v in kinds.items() if v == kind))
    if len(idx) < 200:
        return None
    idx = idx[60:-5]  # skip commander's mode transition at the window start
    thr = np.full(len(d["t"]) + 10, np.nan)
    for k, v in thrust.items():
        thr[int(k)] = v
    n = d["meta"]["n_rotors"]
    S = (d["act"][:, :n] ** 2).sum(axis=1)
    lags = range(5)
    ok = np.all([np.isfinite(thr[idx - L]) for L in lags], axis=0)
    idx = idx[ok]
    X = np.column_stack([np.ones(len(idx))] + [thr[idx - L] for L in lags])
    coef, *_ = np.linalg.lstsq(X, S[idx], rcond=None)
    a, b = coef[0], coef[1:].sum()
    resid = np.array([np.abs(S[idx] - (a + b * thr[idx - L])) / abs(b * 2 * PRBS_AMP)
                      for L in lags])
    worst = [float(r.max()) for r in resid]
    # Per sample, between the two lags that carry the fit: a sample tells them
    # apart where the PRBS differs between them, and the nearer one wins.
    first, second = (int(L) for L in np.argsort(coef[1:])[::-1][:2])
    tells = np.abs(thr[idx - first] - thr[idx - second]) > PRBS_AMP
    nearer = np.where(resid[first] <= resid[second], first, second)[tells]
    per_frame = {f"lag {L}": int((nearer == L).sum()) for L in sorted((first, second))}
    per_frame["of"] = int(tells.sum())
    return coef, worst, per_frame


def step_report(d) -> None:
    t = d["t"]
    att = np.array([euler(q) for q in d["truth_q"]])
    # Truth yaw holds EKF2's heading error as a steady offset from the target,
    # since PX4 controls the estimate; past the settled value removes it.
    print("step: overshoot past the target, past the settled value / settling "
          "(last exit from the band around the target)")
    for start, deg in Steps.yaw_steps:
        prev = [v for s, v in Steps.yaw_steps if s < start]
        size = deg - (prev[-1] if prev else 0.0)
        sel = (t >= start) & (t < start + 15.0)
        yaw = np.degrees(np.unwrap(att[sel, 2]))
        settled = float(yaw[t[sel] >= start + 10.0].mean())
        over = max(0.0, float(((yaw - deg) * np.sign(size)).max()))
        over_settled = max(0.0, float(((yaw - settled) * np.sign(size)).max()))
        out = np.flatnonzero(np.abs(yaw - deg) > 5.0)
        settle = (t[sel][out[-1]] - start) if len(out) else 0.0
        print(f"  yaw {size:+.0f} deg at {start:.0f} s: {over:.2f}, {over_settled:.2f} deg "
              f"/ {settle:.2f} s (settled {settled - deg:+.2f} deg off the target)")
    for start, axis, deg in Steps.attitude_steps:
        i = 0 if axis == "roll" else 1
        sel = (t >= start) & (t < start + 3.0)
        angle = np.degrees(att[sel, i])
        over = max(0.0, float(((angle - deg) * np.sign(deg)).max())) / abs(deg) * 100.0
        out = np.flatnonzero(np.abs(angle - deg) > 0.1 * abs(deg))
        settle = (t[sel][out[-1]] - start) if len(out) else 0.0
        print(f"  {axis} {deg:+.0f} deg at {start:.0f} s: {over:.1f} % / {settle:.2f} s")


def compare(a_path, b_path) -> None:
    a, b = load(a_path), load(b_path)
    n = min(len(a["t"]), len(b["t"]))
    print(f"\n### compare {a_path} {b_path}: {n} samples")
    for key in ("t", "dropped", "proven", "late"):
        same = np.array_equal(a[key][:n], b[key][:n])
        print(f"{key}: {'identical' if same else 'DIFFER'}")
    ages = [np.round((d["t"][:n] - d["est_time"][:n]) / d["meta"]["imu_dt"]) for d in (a, b)]
    both = np.isfinite(ages[0]) & np.isfinite(ages[1])
    first_est = [np.flatnonzero(np.isfinite(x))[:1] for x in ages]
    print(f"first estimate at sample {first_est[0]} vs {first_est[1]}; estimate age "
          f"differs on {int((ages[0][both] != ages[1][both]).sum())} of {int(both.sum())}")
    for key in ("joints", "truth_pos", "act"):
        x, y = a[key][:n], b[key][:n]
        diff = np.abs(x - y).reshape(n, -1).max(axis=1)
        first = np.flatnonzero(diff > 0)
        if len(first) == 0:
            print(f"{key}: bit-identical")
        else:
            i = first[0]
            print(f"{key}: first difference at sample {i} (t={a['t'][i]:.3f} s), "
                  f"max {diff.max():.3g}")
    arm = [np.flatnonzero(d["act_armed"][:n])[:1] for d in (a, b)]
    print(f"first armed sample: {arm[0]} vs {arm[1]}")
    pos = np.linalg.norm(a["truth_pos"][:n] - b["truth_pos"][:n], axis=1)
    print(f"position apart, max {pos.max():.3f} m")


def acceptance(args) -> int:
    """``legs`` at each load, then the checks of AGENTS.md section 3 in one table.
    The thresholds are the decided design: every frame answered after PX4's first
    answer, IMU -> actuator one frame from the frame after the one it arrives in,
    a body-rate setpoint on its frame (lag 2: the setpoint delay plus IMU ->
    actuator), no late estimate."""
    stem = Path(args.out).with_suffix("")
    loads = [int(x) for x in args.loads.split(",")]
    columns = []
    for load in loads:
        args.load, args.out = load, f"{stem}_load{load}.npz"
        print(f"legs, load {load} ...", flush=True)
        path = fly(args)
        if path is None:
            return 1
        columns.append(load_run(path))
    rows: list[tuple[str, list[str], list[bool | None]]] = []

    def row(label, cells, oks=None):
        rows.append((label, cells, oks or [None] * len(cells)))

    tms = [c["timing"] for c in columns]
    cs = [c["counters"] for c in columns]
    row("PX4's first answer, frame", [str(tm["first_answer"]) for tm in tms])
    row("its arrival frame, lag", [str(tm["arrival_lag"]) for tm in tms])
    row("frames unproven after it", [str(c.get("unproven")) for c in cs],
        [c.get("unproven") == 0 for c in cs])
    row("IMU -> actuator one frame", [f"{tm['lags'][1]}/{tm['samples']}" for tm in tms],
        [tm["lags"][1] == tm["samples"] for tm in tms])
    rate = [c["rate"] for c in columns]
    row("body-rate setpoint on its frame",
        [f"{r.get('lag 2', 0)}/{r['of']}" if r else "-" for r in rate],
        [bool(r) and r["of"] > 0 and r.get("lag 2", 0) == r["of"] for r in rate])
    row("late estimates, proven", [str(tm.get("late_proven")) for tm in tms],
        [tm.get("late_proven") == 0 for tm in tms])
    row("estimate age, frames", [",".join(map(str, sorted(tm.get("ages", {})))) for tm in tms])
    att = [c["attitude"] for c in columns]
    row("attitude setpoint, same frame", [f"{a.get('lag 2', 0)}/{a['of']}" if a else "-"
                                          for a in att])
    row("barrier timeouts", [str(c.get("barrier_timeouts")) for c in cs],
        [c.get("barrier_timeouts") == 0 for c in cs])
    row("speed", [f"x{tm['speed']:.1f}" if tm.get("speed") else "-" for tm in tms])

    print(f"\n### acceptance: {Path(args.model).name}")
    print("| | " + " | ".join(f"load {n}" for n in loads) + " |")
    print("|---|" + "---|" * len(loads))
    failed = []
    for label, cells, oks in rows:
        marks = [cell + ("" if ok is None else " PASS" if ok else " FAIL")
                 for cell, ok in zip(cells, oks)]
        print(f"| {label} | " + " | ".join(marks) + " |")
        failed += [label for ok in oks if ok is False]
    print("acceptance: " + ("PASS" if not failed else "FAIL: " + ", ".join(sorted(set(failed)))))
    return 0 if not failed else 1


def load_run(path) -> dict:
    d = load(path)
    rate, att = (setpoint_lags(d, k) for k in (KIND_RATE, KIND_ATTITUDE))
    return dict(timing=timing(d), counters=counters(d["meta"]["summary"]),
                rate=rate[2] if rate else None, attitude=att[2] if att else None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("profile", nargs="?", choices=["legs", "arm-sweep", "steps", "acceptance"],
                   default="legs")
    p.add_argument("--model", type=Path, default=REPO / "models" / "quad_x.xml")
    p.add_argument("--rotors", type=Path, default=None)
    p.add_argument("--airframe", default="22001")
    p.add_argument("--hover", type=float, default=0.2025,
                   help="the airframe's MPC_THR_HOVER (default: the quad's)")
    p.add_argument("-s", "--speed", type=float, default=0.0,
                   help="speed factor; 0, the default, is unpaced")
    p.add_argument("-i", "--instance", type=int, default=0)
    p.add_argument("--load", type=int, default=0, help="processes kept spinning")
    p.add_argument("--loads", default=f"0,{os.cpu_count()}",
                   help="acceptance: the loads to run, comma-separated (default: %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true", help="the simulator's full log")
    p.add_argument("--seed", type=int, default=1, help="thrust PRBS seed")
    p.add_argument("--out", default="/tmp/fly_in_process.npz")
    p.add_argument("--compare", nargs=2, metavar="NPZ")
    p.add_argument("--report", metavar="NPZ")
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return 0
    if args.report:
        report(args.report)
        return 0
    args.period_frames = 1 if args.profile != "arm-sweep" else 5
    setup_logging(args.verbose)
    if args.profile == "acceptance":
        return acceptance(args)
    out = fly(args)
    if out is None:
        return 1
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
