"""The in-process research controller: called synchronously, on simulated time.

A research controller passed to :func:`main.run` runs inside this process, called
from the lockstep loop at sample instants ``t_k = k * period`` of simulated time.
Its command reaches the arm servos at ``t_k + delay`` unless the drop schedule
loses it. Period, delay and drops are *chosen*, not measured, so the age of the
arm command in force is known exactly: at most ``(N + 1) * period + delay`` with
at most ``N`` consecutive drops (:meth:`Schedule.age_bound`). Why the controller
sits here rather than behind the side channel is AGENTS.md section 3.

**What the controller sees is what the vehicle could measure** (:class:`Observation`):
the arm's encoders (:class:`arm.JointReading`, ideal) and PX4's EKF2 estimate as
it arrives over MAVLink (:mod:`px4link`). Ground truth never reaches it. Truth
goes, with the observation and the command, to an optional recorder
(:class:`Sample`), which is where estimation error and data age are measured.

**What stays wall-clock.** Which PX4 estimate has arrived by ``t_k`` (its age is
still exact: it carries PX4's sample time); PX4's own IMU -> actuator response;
and anything the controller sends PX4 itself. None of those is scheduled here.

**Compute time is invisible in simulated time.** The loop stops while the
controller runs, and PX4's clock stops with it, so ``delay`` is where a compute
budget goes. Each call's wall time is recorded, and handed back to the loop so
its pacer does not follow a slow call with a catch-up burst -- which would widen
PX4's lead, the one delay here nobody chose.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arm import ArmCommand, ArmStatus, JointReading
from .frames import GeodeticProjection, rebase_ned
from .px4link import EstimateLink, Px4Estimate
from .sim import Physics, SimState

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """What the controller gets at sample ``index``."""

    index: int
    time: float  # the sample instant t_k, simulated seconds
    joints: JointReading  # read at t_k
    # The newest EKF2 estimate that has arrived by t_k. None until PX4 streams
    # one, which is not before PX4 has booted: t_k starts at 0, PX4 does not.
    estimate: Px4Estimate | None

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


@dataclass(frozen=True)
class Sample:
    """One sample instant, for the recorder."""

    obs: Observation
    truth: Truth
    command: NDArray[np.float64] | None  # what the controller returned
    dropped: bool  # lost by the drop schedule; only meaningful with a command
    apply_time: float | None  # when it reaches the servos; None if it never will
    compute_wall_s: float  # wall seconds the controller took


class Controller(Protocol):
    def step(self, obs: Observation) -> ArrayLike | None:
        """Arm joint targets (rad, one per ``arm_act*`` servo), or None to send
        nothing this sample. Must not block on I/O: the whole simulation, PX4's
        clock included, waits for it."""
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


@dataclass(frozen=True)
class Schedule:
    """When the controller runs, and what happens to its commands.

    ``period`` and ``delay`` are simulated seconds, whole multiples of the IMU
    frame; a delay of 0 drives the frame starting at ``t_k`` itself. ``drops``
    maps a sample index to "lost?" and is called once per sample, in order; its
    ``max_consecutive`` attribute, if it has one, bounds the command age.
    """

    period: float
    delay: float = 0.0
    drops: Callable[[int], bool] | None = None

    def frames(self, imu_dt: float) -> tuple[int, int]:
        """``(period, delay)`` in IMU frames; raises unless both are whole."""
        out = []
        for name, value, least in (("period", self.period, 1), ("delay", self.delay, 0)):
            n = int(round(value / imu_dt))
            if n < least or abs(n * imu_dt - value) > 1e-9:
                raise ValueError(
                    f"controller {name} {value!r} s is not a whole number (>= {least}) "
                    f"of IMU frames of {imu_dt:g} s"
                )
            out.append(n)
        return out[0], out[1]

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


class ControllerHost:
    """Runs a :class:`Controller` from the lockstep loop, once per IMU frame."""

    def __init__(
        self,
        controller: Controller,
        schedule: Schedule,
        imu_dt: float,
        *,
        estimates: EstimateLink | None = None,
        home: GeodeticProjection | None = None,
        recorder: Callable[[Sample], None] | None = None,
    ) -> None:
        self.controller = controller
        self.schedule = schedule
        self.imu_dt = imu_dt
        self.period_frames, self.delay_frames = schedule.frames(imu_dt)
        self.estimates = estimates
        self.home = home
        self.recorder = recorder

        self.frame = 0
        self.samples = 0
        self.commands = 0
        self.dropped = 0
        self.idle = 0
        self.compute_total = 0.0
        self.compute_max = 0.0
        self.max_estimate_age: float | None = None
        self._pending: deque[tuple[int, ArmCommand]] = deque()

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
        )

    def before_step(self, physics: Physics) -> float:
        """Call once per frame, before ``step_frame``. Returns the wall seconds
        spent in the controller, which the loop's pacer must not count."""
        now = physics.time
        estimate = None if self.estimates is None else self.estimates.poll(now)
        compute = 0.0
        if self.frame % self.period_frames == 0:
            compute = self._sample(physics, now, estimate)
        while self._pending and self._pending[0][0] <= self.frame:
            physics.submit_arm_command(self._pending.popleft()[1])
        self.frame += 1
        return compute

    def _sample(self, physics: Physics, now: float, estimate: Px4Estimate | None) -> float:
        index = self.samples
        self.samples += 1
        joints = physics.arm_joints()
        obs = Observation(index=index, time=now, joints=joints, estimate=estimate)
        if obs.estimate_age is not None:
            self.max_estimate_age = max(self.max_estimate_age or 0.0, obs.estimate_age)

        start = time.perf_counter()
        raw = self.controller.step(obs)
        compute = time.perf_counter() - start
        self.compute_total += compute
        self.compute_max = max(self.compute_max, compute)

        dropped = False if self.schedule.drops is None else bool(self.schedule.drops(index))
        command = None if raw is None else np.asarray(raw, dtype=np.float64)
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

        if self.recorder is not None:
            self.recorder(Sample(
                obs=obs, truth=self._truth(physics, joints), command=command,
                dropped=dropped, apply_time=apply_time, compute_wall_s=compute,
            ))
        return compute

    def _truth(self, physics: Physics, joints: JointReading) -> Truth:
        state = physics.state()
        origin = None if self.estimates is None else self.estimates.origin
        pos_ekf = (
            None if origin is None or self.home is None
            else rebase_ned(state.pos_ned, self.home, origin)
        )
        return Truth(state=state, joints=joints, arm=physics.arm_status(), pos_ned_ekf=pos_ekf)

    def summary(self) -> str:
        mean = self.compute_total / self.samples if self.samples else 0.0
        age = "-" if self.max_estimate_age is None else f"{self.max_estimate_age * 1e3:.1f}ms"
        return (
            f"ctrl samples={self.samples} cmd={self.commands} dropped={self.dropped} "
            f"idle={self.idle} compute={mean * 1e3:.2f}/{self.compute_max * 1e3:.2f}ms "
            f"est_age_max={age}"
        )
