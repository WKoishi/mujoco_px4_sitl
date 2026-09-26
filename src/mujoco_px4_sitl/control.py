"""The in-process research controller: called synchronously, on simulated time.

A research controller passed to :func:`main.run` runs inside this process, called
from the lockstep loop at sample instants ``t_k = k * period`` of simulated time.
Every leg between it and the vehicle is *chosen*, not measured (AGENTS.md
section 3 has why it sits here):

* **The arm**: its command reaches the servos at ``t_k + delay`` unless the drop
  schedule loses it, so the age of the arm command in force is at most
  ``(N + 1) * period + delay`` with at most ``N`` consecutive drops
  (:meth:`Schedule.age_bound`).
* **PX4's estimate**: the observation carries the newest EKF2 output with sample
  time at or before ``t_k - estimate_delay``, at least one frame. EKF2 publishes
  every other frame, so its age is that delay or one frame more. The link fetches
  every output after each frame PX4 has answered (:mod:`px4link`); one that
  arrives only after ``t_k`` marks the sample late in its record.
* **PX4 setpoints and commands**: what the controller returns at ``t_k`` is in
  uORB before PX4 processes the frame at ``t_k + setpoint_delay``, at least one
  frame, behind a barrier. Arming and mode changes too, so a run is armed at a
  simulated time.
* **PX4's own IMU -> actuator response** is one frame, by the loop (:mod:`loop`),
  except on the boot frame PX4's first answer arrives in.

Those guarantees hold on frames PX4 answered in time. A sample on any other
frame is marked unproven (:attr:`Observation.proven`), never silently.

**What the controller sees is what the vehicle could measure** (:class:`Observation`):
the arm's encoders (:class:`arm.JointReading`, ideal), PX4's EKF2 estimate as it
arrives over MAVLink, and PX4's command acknowledgements. Ground truth never
reaches it. Truth goes, with the observation and the outputs, to an optional
recorder (:class:`Sample`), which is where estimation error and data age are
measured.

**Compute time is invisible in simulated time.** The loop stops while the
controller runs, and PX4's clock stops with it, so the delays are where a
compute budget goes. Each call's wall time is recorded, and handed back to the
loop so its pacer does not follow a slow call with a catch-up burst.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymavlink.dialects.v20 import common as mavlink

from .arm import ArmCommand, ArmStatus, JointReading
from .frames import GeodeticProjection, rebase_ned
from .hil import ActuatorControls
from .px4link import ApiLink, CommandAck, Px4Estimate, check_barrier_safe
from .sim import Physics, SimState

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """What the controller gets at sample ``index``."""

    index: int
    time: float  # the sample instant t_k, simulated seconds
    joints: JointReading  # read at t_k
    # The newest EKF2 estimate received with sample time at or before
    # t_k - estimate_delay. None until PX4 sends one, which is not before PX4
    # has booted: t_k starts at 0, PX4 does not.
    estimate: Px4Estimate | None
    # PX4 answered this frame in time, and every EKF2 output the estimate could
    # be was fetched behind an echoed barrier, so the timing guarantees hold.
    # False before PX4's first answer, until the API link has fetched for
    # estimate_delay + 2 frames in a row, and after a frame that broke either.
    proven: bool = False
    # COMMAND_ACKs from PX4 since the previous sample, oldest first.
    acks: tuple[CommandAck, ...] = ()

    @property
    def estimate_age(self) -> float | None:
        """Simulated seconds from EKF2's sample to ``time``. Exact."""
        return None if self.estimate is None else self.time - self.estimate.time


@dataclass(frozen=True)
class Truth:
    """Ground truth at a sample instant. For the recorder only."""

    state: SimState  # the side channel's frames: pos_ned relative to the home
    joints: JointReading
    arm: ArmStatus | None  # cmd_age is the age of the command in force at t_k
    # The truth position in EKF2's local frame, once PX4 has sent its origin:
    # what Px4Estimate.pos_ned should be compared with. None before that.
    pos_ned_ekf: NDArray[np.float64] | None
    # PX4's answer that drives the rotors over [t_k, t_k+1); its time_usec is
    # the frame it answers, t_k-1 when PX4 answered in time.
    actuators: ActuatorControls | None = None


@dataclass(frozen=True)
class Sample:
    """One sample instant, for the recorder."""

    obs: Observation
    truth: Truth
    command: NDArray[np.float64] | None  # the arm targets the controller returned
    dropped: bool  # lost by the drop schedule; only meaningful with a command
    apply_time: float | None  # when it reaches the servos; None if it never will
    compute_wall_s: float  # wall seconds the controller took
    # What the controller returned for PX4, and the frame PX4 processes with it.
    px4: tuple[mavlink.MAVLink_message, ...] = ()
    px4_time: float | None = None
    # An estimate the observation should have carried arrived only after t_k.
    estimate_late: bool = False


@dataclass(frozen=True)
class ControllerOutput:
    """What a controller may return besides bare arm targets.

    ``px4`` holds MAVLink messages for PX4's API link, in order: setpoints and
    commands, built with :mod:`px4link`'s helpers or from pymavlink's
    ``common`` dialect with target ids left 0. Requests that would block PX4's
    receive thread are refused (:func:`px4link.check_barrier_safe`).
    """

    arm: ArrayLike | None = None
    px4: Sequence[mavlink.MAVLink_message] = field(default_factory=tuple)


class Controller(Protocol):
    def step(self, obs: Observation) -> ArrayLike | ControllerOutput | None:
        """Arm joint targets (rad, one per ``arm_act*`` servo), a
        :class:`ControllerOutput`, or None to send nothing this sample. Must not
        block on I/O: the whole simulation, PX4's clock included, waits for it."""
        ...


class CappedDrops:
    """Loses each command with probability ``p``, never more than
    ``max_consecutive`` in a row. Seeded, so a run repeats exactly.

    Called once per sample instant, in order, whether or not the controller
    produced a command -- so the pattern does not depend on what it returned.
    """

    def __init__(self, p: float, max_consecutive: int, seed: int = 0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("drop probability must be in [0, 1]")
        if max_consecutive < 0:
            raise ValueError("max_consecutive must be >= 0")
        self.p = float(p)
        self.max_consecutive = int(max_consecutive)
        self._rng = np.random.default_rng(seed)
        self._run = 0

    def __call__(self, index: int) -> bool:  # noqa: ARG002 - the order is the index
        # Draw every time, even when the cap forces the answer, so the stream of
        # draws -- and every later decision -- does not depend on the cap.
        drop = bool(self._rng.random() < self.p) and self._run < self.max_consecutive
        self._run = self._run + 1 if drop else 0
        return drop


class ScheduleFrames(NamedTuple):
    period: int
    delay: int
    estimate_delay: int
    setpoint_delay: int


@dataclass(frozen=True)
class Schedule:
    """When the controller runs, and the delay on each leg.

    All times are simulated seconds, whole multiples of the IMU frame.
    ``delay`` takes an arm command to the servos; 0 drives the frame starting at
    ``t_k`` itself. ``drops`` maps a sample index to "lost?" and is called once
    per sample, in order; its ``max_consecutive`` attribute, if it has one,
    bounds the arm command's age. Drops apply to arm commands only.

    ``estimate_delay`` and ``setpoint_delay`` are PX4's legs, at least one frame
    each; None means one. An estimate of age 0 cannot be proven complete, and a
    setpoint must be in before the frame PX4 processes with it
    (IMPLEMENTATION_PLAN.md 3.9).
    """

    period: float
    delay: float = 0.0
    drops: Callable[[int], bool] | None = None
    estimate_delay: float | None = None
    setpoint_delay: float | None = None

    def frames(self, imu_dt: float) -> ScheduleFrames:
        """Every time in IMU frames; raises unless each is whole and in range."""
        out = []
        for name, value, least in (
            ("period", self.period, 1), ("delay", self.delay, 0),
            ("estimate_delay", self.estimate_delay, 1),
            ("setpoint_delay", self.setpoint_delay, 1),
        ):
            if value is None:
                out.append(least)
                continue
            n = int(round(value / imu_dt))
            if n < least or abs(n * imu_dt - value) > 1e-9:
                raise ValueError(
                    f"controller {name} {value!r} s is not a whole number (>= {least}) "
                    f"of IMU frames of {imu_dt:g} s"
                )
            out.append(n)
        return ScheduleFrames(*out)

    @property
    def max_consecutive_drops(self) -> int | None:
        if self.drops is None:
            return 0
        cap = getattr(self.drops, "max_consecutive", None)
        return None if cap is None else int(cap)

    def age_bound(self) -> float | None:
        """Largest age, over continuous time, of the arm command in force: the
        H of ``H <= (N + 1) h + tau``. Holds while the controller returns a
        command every sample; ``cmd_age``, read at frame starts, peaks one frame
        below it. None if the drop schedule states no cap."""
        cap = self.max_consecutive_drops
        return None if cap is None else (cap + 1) * self.period + self.delay


@dataclass
class _Pending:
    """A sample held back until its estimate's lateness is decided."""

    sample: Sample
    cutoff: float  # t_k - estimate_delay
    given: float | None  # sample time of the estimate the observation carried
    late: bool = False


class ControllerHost:
    """Runs a :class:`Controller` from the lockstep loop, once per IMU frame.

    The loop calls :meth:`before_sensor` before each frame's ``HIL_SENSOR``,
    :meth:`before_step` once PX4 has answered it (or the fallback has given up),
    and :meth:`close` when it stops.

    A sample reaches the recorder once whether its estimate was late is decided:
    when an estimate newer than its cutoff has arrived, since the link reads them
    in order, or at :meth:`close`.
    """

    def __init__(
        self,
        controller: Controller,
        schedule: Schedule,
        imu_dt: float,
        *,
        link: ApiLink | None = None,
        home: GeodeticProjection | None = None,
        recorder: Callable[[Sample], None] | None = None,
    ) -> None:
        self.controller = controller
        self.schedule = schedule
        self.imu_dt = imu_dt
        frames = schedule.frames(imu_dt)
        self.period_frames = frames.period
        self.delay_frames = frames.delay
        self.estimate_delay_frames = frames.estimate_delay
        self.setpoint_delay_frames = frames.setpoint_delay
        # EKF2 publishes every other frame at its default EKF2_PREDICT_US, so an
        # estimate is the delay old or one frame older.
        self.estimate_age_bound = (self.estimate_delay_frames + 1) * imu_dt
        self.link = link
        self.home = home
        self.recorder = recorder

        self.frame = 0
        self.samples = 0
        self.commands = 0
        self.dropped = 0
        self.idle = 0
        self.unproven = 0
        self.late = 0
        self.over_age = 0
        self.px4_sends = 0
        self.px4_unproven = 0
        self.compute_total = 0.0
        self.compute_max = 0.0
        self.max_estimate_age: float | None = None
        self._pending: deque[tuple[int, ArmCommand]] = deque()
        self._px4_pending: deque[tuple[int, list[mavlink.MAVLink_message]]] = deque()
        self._records: deque[_Pending] = deque()
        self._estimates_seen = 0
        self._acks_seen = 0
        # Frames in a row, ending now, whose answer came and whose fetch was
        # echoed. The output an estimate needs, at or before t_k - d, is out by
        # frame k - d + 1 even under load, and one fetch takes only the newest,
        # so frames k - d - 1 .. k must all have fetched (plan 3.9).
        self._fetch_run = 0

    def describe(self, arm_timeout: float) -> str:
        """One line for the log, and a warning if the watchdog would trip within
        the schedule's own age bound."""
        bound = self.schedule.age_bound()
        cap = self.schedule.max_consecutive_drops
        if bound is not None and 0.0 < arm_timeout < bound:
            _log.warning(
                "arm watchdog timeout %.3f s is below the schedule's command-age "
                "bound %.3f s: the watchdog will act on drops the schedule allows",
                arm_timeout, bound,
            )
        return (
            f"controller every {self.schedule.period:g} s, delay {self.schedule.delay:g} s, "
            + ("no drops" if self.schedule.drops is None else
               f"drops <= {cap} in a row" if cap is not None else "uncapped drops")
            + (f"; arm command age <= {bound:g} s" if bound is not None else "")
            + f"; estimate delay {self.estimate_delay_frames * self.imu_dt:g} s, "
            f"setpoint delay {self.setpoint_delay_frames * self.imu_dt:g} s"
        )

    def before_sensor(self, now: float) -> None:
        """Call before the ``HIL_SENSOR`` of the frame at ``now``: sends what is
        due at this frame, behind a barrier, so PX4 processes the frame with it."""
        due: list[mavlink.MAVLink_message] = []
        while self._px4_pending and self._px4_pending[0][0] <= self.frame:
            due.extend(self._px4_pending.popleft()[1])
        if not due:
            return
        self.px4_sends += 1
        if not self.link.send(due, now):
            self.px4_unproven += 1

    def before_step(
        self,
        physics: Physics,
        answered: bool = False,
        actuators: ActuatorControls | None = None,
    ) -> float:
        """Call once per frame, before ``step_frame``. ``answered`` says PX4's
        answer stamped with this frame arrived in time; ``actuators`` is the
        answer that drives this frame. Returns the wall seconds spent in the
        controller, which the loop's pacer must not count."""
        now = physics.time
        if self.link is None:
            proven = answered
        else:
            self.link.poll(now)
            fetched = answered and self.link.fetch(now)
            self._fetch_run = self._fetch_run + 1 if fetched else 0
            proven = self._fetch_run >= self.estimate_delay_frames + 2
            self._decide()
        compute = 0.0
        if self.frame % self.period_frames == 0:
            compute = self._sample(physics, now, proven, actuators)
        while self._pending and self._pending[0][0] <= self.frame:
            physics.submit_arm_command(self._pending.popleft()[1])
        self.frame += 1
        return compute

    def _sample(
        self, physics: Physics, now: float, proven: bool, actuators: ActuatorControls | None
    ) -> float:
        index = self.samples
        self.samples += 1
        joints = physics.arm_joints()
        cutoff = now - self.estimate_delay_frames * self.imu_dt
        estimate = acks = None
        if self.link is not None:
            estimate = self.link.estimate_at(cutoff)
            acks = tuple(self.link.acks[self._acks_seen:])
            self._acks_seen = len(self.link.acks)
        obs = Observation(index=index, time=now, joints=joints, estimate=estimate,
                          proven=proven, acks=acks or ())
        if not proven:
            self.unproven += 1
        age = obs.estimate_age
        if age is not None:
            self.max_estimate_age = max(self.max_estimate_age or 0.0, age)
            if proven and age > self.estimate_age_bound + 1e-9:
                self.over_age += 1

        start = time.perf_counter()
        raw = self.controller.step(obs)
        compute = time.perf_counter() - start
        self.compute_total += compute
        self.compute_max = max(self.compute_max, compute)

        if isinstance(raw, ControllerOutput):
            arm, px4 = raw.arm, list(raw.px4)
        else:
            arm, px4 = raw, []
        dropped = False if self.schedule.drops is None else bool(self.schedule.drops(index))
        command = None if arm is None else np.asarray(arm, dtype=np.float64)
        apply_time = None
        if command is None:
            self.idle += 1
        else:
            self.commands += 1
            if dropped:
                self.dropped += 1
            else:
                # state_time is the joint sample instant, so the servos' cmd_age
                # is the age of the command path. The estimate's own age is on
                # top of that, and in the Sample.
                self._pending.append((
                    self.frame + self.delay_frames,
                    ArmCommand(seq=index, mode="position", values=command, state_time=now),
                ))
                apply_time = now + self.delay_frames * self.imu_dt
        px4_time = None
        if px4:
            if self.link is None:
                raise RuntimeError("the controller returned PX4 messages, but there "
                                   "is no API link to send them on")
            for msg in px4:
                check_barrier_safe(msg)
            self._px4_pending.append((self.frame + self.setpoint_delay_frames, px4))
            px4_time = now + self.setpoint_delay_frames * self.imu_dt

        if self.recorder is not None:
            sample = Sample(
                obs=obs, truth=self._truth(physics, joints, actuators), command=command,
                dropped=dropped, apply_time=apply_time, compute_wall_s=compute,
                px4=tuple(px4), px4_time=px4_time,
            )
            self._records.append(_Pending(
                sample, cutoff, None if estimate is None else estimate.time
            ))
            self._decide()
        return compute

    def _decide(self) -> None:
        """Mark late samples, and pass on those whose lateness is decided."""
        if not self._records:
            return
        latest = None if self.link is None else self.link.latest
        if self.link is not None and self.link.received != self._estimates_seen:
            self._estimates_seen = self.link.received
            for record in self._records:
                if record.late:
                    continue
                best = self.link.estimate_at(record.cutoff)
                if best is not None and (record.given is None or best.time > record.given):
                    record.late = True
                    if record.sample.obs.proven:
                        self.late += 1
        while self._records and (
            self.link is None or (latest is not None and latest.time > self._records[0].cutoff)
        ):
            self._deliver(self._records.popleft())

    def _deliver(self, record: _Pending) -> None:
        self.recorder(replace(record.sample, estimate_late=record.late))

    def close(self) -> None:
        """Pass on every sample still held back; the loop calls this at its end."""
        self._decide()
        while self._records:
            self._deliver(self._records.popleft())

    def _truth(
        self, physics: Physics, joints: JointReading, actuators: ActuatorControls | None
    ) -> Truth:
        state = physics.state()
        origin = None if self.link is None else self.link.origin
        pos_ekf = (
            None if origin is None or self.home is None
            else rebase_ned(state.pos_ned, self.home, origin)
        )
        return Truth(state=state, joints=joints, arm=physics.arm_status(),
                     pos_ned_ekf=pos_ekf, actuators=actuators)

    def summary(self) -> str:
        mean = self.compute_total / self.samples if self.samples else 0.0
        age = "-" if self.max_estimate_age is None else f"{self.max_estimate_age * 1e3:.1f}ms"
        line = (
            f"ctrl samples={self.samples} unproven={self.unproven} cmd={self.commands} "
            f"dropped={self.dropped} idle={self.idle} "
            f"compute={mean * 1e3:.2f}/{self.compute_max * 1e3:.2f}ms "
            f"est_age_max={age} late={self.late} over_age={self.over_age} "
            f"px4_sends={self.px4_sends} px4_unproven={self.px4_unproven}"
        )
        return line if self.link is None else f"{line} {self.link.summary()}"
