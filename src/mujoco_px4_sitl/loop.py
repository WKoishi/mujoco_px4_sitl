"""Lockstep orchestration: strict lockstep, and the plan section 3.2 fallback.

**Once PX4 has answered, every frame waits for the answer stamped with its
time.** The frame ``[t_k, t_k+1)`` is driven by PX4's answer to the
``HIL_SENSOR`` stamped ``t_k-1``, so IMU -> actuator is exactly one frame, and
``HIL_SENSOR(t_k+1)`` is not sent before the answer stamped ``t_k`` has arrived.
The stamp is PX4's clock when it sent the answer, which only our ``HIL_SENSOR``
moves, so with no frame outstanding it proves which frame the answer was
computed from (plan 3.2, "What an answer proves").

**Before PX4's first answer, and after a frame whose answer timed out, the
bounded-lead loop runs**: wall clock paces it, and the actuator stream only
bounds how far ahead of PX4 it may get. Both halves are mandatory there:

* Without the pacer, nothing bounds us during PX4's boot -- which happens
  entirely inside this loop's first frames, because ``simulator_mavlink start``
  blocks ``rcS`` until our first ``HIL_SENSOR``. Running ahead drops IMU FIFO
  samples and presents as estimator divergence.
* Without the brake, a PX4 that fell behind is outrun the same way.
* Blocking on ``HIL_ACTUATOR_CONTROLS`` *from the first frame* deadlocks at
  startup: PX4 publishes none before Commander is up, and its escape hatches are
  either measured in simulated time or compiled out under lockstep.

Every timeout here is **wall clock**, and none is fatal. A frame whose answer
timed out is counted as unproven, never silently.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pymavlink.dialects.v20 import common as mavlink

from . import hil
from .config import Config
from .control import ControllerHost
from .sim import Physics
from .sidechannel import SideChannel
from .transport import HilServer

_log = logging.getLogger(__name__)


@dataclass
class LoopStats:
    """Counters that make the regime and the section 7 loop faults visible."""

    frames: int = 0
    actuator_messages: int = 0
    # Of the frames from the one PX4's first answer arrived in: those whose own
    # answer, the one stamped with their time, arrived before the next frame
    # was sent, and those it did not arrive for within answer_timeout_s. The
    # two add up to every frame after the boot, frames - answered - unproven.
    answered: int = 0
    unproven: int = 0
    brake_waits: int = 0
    brake_timeouts: int = 0
    discarded_messages: int = 0
    # Sim/wall time spent *inside* the loop, excluding connection setup. The
    # ratio of the two is the section 7 diagnostic: it should sit near
    # speed_factor, and neither grow without bound (we outran PX4, IMU FIFO
    # samples dropped) nor collapse toward zero (the brake is pacing the loop).
    sim_time: float = 0.0
    wall_time: float = 0.0
    # How many IMU frames old PX4's controls were when a frame used them, from
    # HIL_ACTUATOR_CONTROLS.time_usec -- PX4's clock when it sent them. One on
    # every frame after an answered one, by construction; more only after an
    # unproven frame or in the boot.
    px4_lag: Counter = field(default_factory=Counter)

    @property
    def ratio(self) -> float:
        return (self.sim_time / self.wall_time) if self.wall_time > 0.0 else 0.0

    def summary(self) -> str:
        return (
            f"t_sim={self.sim_time:8.2f}s ratio={self.ratio:5.3f} "
            f"frames={self.frames} act={self.actuator_messages} "
            f"answered={self.answered} unproven={self.unproven} "
            f"brake={self.brake_waits} timeouts={self.brake_timeouts}"
        )


class LockstepLoop:
    """Drives physics, the HIL link, and the side channel."""

    def __init__(
        self,
        cfg: Config,
        physics: Physics,
        server: HilServer,
        sidechannel: SideChannel | None = None,
        frame_hook: Callable[[], None] | None = None,
        controller: ControllerHost | None = None,
    ) -> None:
        self.cfg = cfg
        self.physics = physics
        self.server = server
        self.sidechannel = sidechannel
        # Called once per IMU frame. Used for the viewer, which must never gate
        # the physics loop, so it decimates internally.
        self.frame_hook = frame_hook
        # The in-process research controller, run synchronously every frame
        # (control.py). With one attached, side-channel arm_cmd is refused: two
        # writers would take turns on the servos by arrival order.
        self.controller = controller
        self._refused_arm_cmds = 0
        self.stats = LoopStats()
        # The newest answer received, whatever its stamp; for the flags.
        self.controls = hil.ActuatorControls()
        self.running = False

        # What drives the frame being stepped: the newest answer stamped at or
        # before the previous frame's time, or none yet (stamp -1: an answer to
        # the frame at t = 0 is stamped 0). And the current frame's own answer,
        # which drives the next one.
        self._held = hil.ActuatorControls(time_usec=-1)
        self._own: hil.ActuatorControls | None = None
        # Strict once any answer has arrived, until a frame's answer times out.
        self._strict = False
        self.first_answer_frame: int | None = None
        self._frames_since_ack = 0
        self._imu_time_us = 0
        self._last_imu_time_us = -1
        self._lockstep_checked = False
        self._sidechannel_decimation = max(
            1, int(round(cfg.imu_rate_hz / max(1e-6, cfg.sidechannel_rate_hz)))
        )

    # -- inbound ------------------------------------------------------------

    def _handle(self, msg: mavlink.MAVLink_message) -> bool:
        """Returns True if this was a fresh HIL_ACTUATOR_CONTROLS."""
        msg_type = msg.get_type()
        if msg_type == "HIL_ACTUATOR_CONTROLS":
            self.controls = hil.decode_actuator_controls(msg)
            self.stats.actuator_messages += 1
            self._take(self.controls)
            if not self._lockstep_checked:
                self._lockstep_checked = True
                if self.controls.lockstep:
                    _log.info("PX4 confirms lockstep build (flags bit 0 set)")
                else:
                    _log.warning(
                        "PX4 did NOT set the lockstep flag: this is a nolockstep "
                        "build, so PX4 does not take its clock from us. The IMU "
                        "cadence must be paced against wall clock instead."
                    )
            return True
        # PX4 sends an unsolicited HEARTBEAT and a COMMAND_LONG
        # (SET_MESSAGE_INTERVAL for HIL_STATE_QUATERNION at 200 Hz) right after
        # connecting, in no guaranteed order. Neither needs a reply and nothing
        # may be gated on either arriving (plan 3.1). Discard quietly.
        self.stats.discarded_messages += 1
        if msg_type == "BAD_DATA":
            _log.debug("discarding malformed frame")
        return False

    def _take(self, controls: hil.ActuatorControls) -> None:
        """Route an answer by its stamp. Every answer, on every path, zeroes the
        lead (plan 3.2's invariant) and makes the next frame strict."""
        self._frames_since_ack = 0
        if not self._strict:
            self._strict = True
            if self.first_answer_frame is None:
                self.first_answer_frame = self.stats.frames
                _log.info("PX4's first answer, frame %d: strict lockstep from here",
                          self.stats.frames)
            elif self.stats.unproven <= 3 or self.stats.unproven % 100 == 0:
                _log.info("PX4 answering again, frame %d: strict lockstep resumes",
                          self.stats.frames)
        stamp = controls.time_usec
        if stamp == self._imu_time_us:
            self._own = controls
        elif self._held.time_usec < stamp < self._imu_time_us:
            # Another frame's answer is never this frame's. An older one late
            # after a timeout, or one the boot's lead left behind, is still the
            # newest at or before the previous frame, so it drives this frame.
            self._held = controls
        # A stamp ahead of our clock cannot come from PX4, whose clock is ours.

    def _await_answer(self) -> bool:
        """Strict: wait, wall clock, for the answer stamped with this frame."""
        deadline = time.monotonic() + self.cfg.answer_timeout_s
        while self._own is None and self.server.connected:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            for msg in self.server.wait(remaining):
                self._handle(msg)
        return self._own is not None

    def _drain(self) -> None:
        """Take everything readable and, on a fresh actuator message, clear the
        lead.

        Resetting here is what makes the brake a *brake*. Without it the counter
        only ever falls in :meth:`_brake`, so braking fires every
        ``max_lead_frames`` frames on a fixed cadence no matter how promptly PX4
        replies -- the "brake is pacing the loop" fault of plan 7, and invisible
        at ``speed_factor = 1.0`` because the pacer's own sleep absorbs the cost.
        """
        got = False
        for msg in self.server.drain():
            got = self._handle(msg) or got
        if got:
            self._frames_since_ack = 0

    def _brake(self) -> None:
        """Bounded lead exceeded: wait for PX4, but never forever."""
        self.stats.brake_waits += 1
        got = False
        for msg in self.server.wait(self.cfg.brake_timeout_s):
            got = self._handle(msg) or got
        if got:
            self._frames_since_ack = 0
            return
        # Timeout. PX4 is quiet (disarmed, mode transition), not slow. Hand
        # pacing back to the wall clock and keep the last-known-good controls:
        # they are a held setpoint. Never fatal.
        self.stats.brake_timeouts += 1
        self._frames_since_ack = 0
        if self.stats.brake_timeouts % 20 == 1:
            _log.info(
                "brake timeout (%d total): no HIL_ACTUATOR_CONTROLS within %.0f ms "
                "wall clock; holding previous controls",
                self.stats.brake_timeouts, self.cfg.brake_timeout_s * 1e3,
            )

    # -- outbound -----------------------------------------------------------

    def _send_frame(self) -> bool:
        state = self.physics.state()
        # Our IMU timestamp *is* PX4's clock under lockstep, so monotonicity is
        # a correctness requirement, not a nicety (plan 3.2 / 7.3).
        self._imu_time_us = int(round(state.time * 1e6))
        if self._imu_time_us <= self._last_imu_time_us:
            raise RuntimeError(
                f"IMU timestamp not strictly monotonic: {self._imu_time_us} us "
                f"after {self._last_imu_time_us} us -- this is PX4's clock"
            )
        self._last_imu_time_us = self._imu_time_us

        mav = self.server.mav
        if not self.server.send(
            hil.encode_hil_sensor(mav, self._imu_time_us, state.accel_frd, state.gyro_frd)
        ):
            return False
        # Every IMU frame: 250 Hz, above the 200 Hz PX4 asks for, which is not an
        # integer divisor of the IMU rate. Nothing in PX4 checks the interval.
        return self.server.send(
            hil.encode_hil_state_quaternion(
                mav, self._imu_time_us, state.q_px4, state.gyro_frd,
                state.lat_deg, state.lon_deg, state.alt_m, state.vel_ned,
                state.accel_frd,
            )
        )

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        self.running = True
        if not self.server.connected:
            _log.info("waiting for PX4 to connect (it retries until we accept)")
            # Poll rather than block indefinitely, so stop() is honoured here
            # too. PX4 may never boot at all -- a bad airframe id is enough --
            # and a wait that ignores SIGINT/SIGTERM hangs run_sitl.sh's
            # cleanup, which signals and then waits on us.
            while self.running and not self.server.accept(timeout=0.5):
                pass
            if not self.running:
                _log.info("stopped before PX4 connected")
                return

        frame_wall_dt = cfg.imu_dt / cfg.speed_factor if cfg.speed_factor > 0.0 else None
        t_wall_start = time.monotonic()
        t_sim_start = self.physics.time
        t_wall_next = t_wall_start
        t_status_next = t_wall_start + cfg.status_interval_s
        controls = np.zeros(self.physics.num_actuators)

        _log.info(
            "loop start: IMU %.0f Hz, speed %s, answer timeout %.0f ms; fallback "
            "max_lead=%d frames, brake timeout %.0f ms",
            cfg.imu_rate_hz,
            f"x{cfg.speed_factor:.2f}" if frame_wall_dt else "unpaced",
            cfg.answer_timeout_s * 1e3, cfg.max_lead_frames, cfg.brake_timeout_s * 1e3,
        )

        while self.running:
            if self.controller is not None:
                # Setpoints due at this frame go in before its HIL_SENSOR, so
                # PX4 processes the frame with them.
                self.controller.before_sensor(self.physics.time)
            if not self._send_frame():
                _log.warning("PX4 link lost, stopping loop")
                break
            self._frames_since_ack += 1

            if not self._strict:
                self._drain()
                if self._frames_since_ack >= cfg.max_lead_frames:
                    self._brake()
            # Strict already, or an answer has just arrived -- PX4's first, or
            # one after a timeout. Either way PX4 now answers every frame
            # (plan 3.2), so this one waits for its own too.
            answered = False
            if self._strict:
                answered = self._await_answer()
                if not answered:
                    # Never fatal: the loop falls back to the pacer and the
                    # brake, and the next answer makes it strict again.
                    self._strict = False
                    if self.stats.unproven < 3 or self.stats.unproven % 100 == 99:
                        _log.warning(
                            "no answer stamped %d us within %.0f ms wall clock: "
                            "frame unproven (%d so far), falling back until PX4 "
                            "answers again", self._imu_time_us,
                            cfg.answer_timeout_s * 1e3, self.stats.unproven + 1,
                        )
            if self.first_answer_frame is not None:
                if answered:
                    self.stats.answered += 1
                else:
                    self.stats.unproven += 1

            # Every frame and just before stepping, so an arm_cmd drives the
            # first frame after it arrives rather than waiting for a publish.
            if self.sidechannel is not None:
                for command in self.sidechannel.poll():
                    if self.controller is None:
                        self.physics.submit_arm_command(command)
                    else:
                        self._refuse(command.seq)
            if self.controller is not None:
                # The loop was stopped while the controller ran, and PX4's clock
                # with it. Not counting that time keeps the pacer from following
                # a slow call with a catch-up burst.
                t_wall_next += self.controller.before_step(
                    self.physics, answered=answered,
                    actuators=self._held if self._held.time_usec >= 0 else None,
                )

            if self._held.time_usec >= 0:
                lag = (self._imu_time_us - self._held.time_usec) * 1e-6 / cfg.imu_dt
                self.stats.px4_lag[int(round(lag))] += 1
            controls = self._held.effective(self.physics.num_actuators)
            self.physics.step_frame(controls)
            self.stats.frames += 1
            if self._own is not None:
                self._held, self._own = self._own, None

            if self.sidechannel is not None and self.stats.frames % self._sidechannel_decimation == 0:
                self.sidechannel.publish(
                    self.physics.state(), controls, self.physics.arm_status()
                )

            if self.frame_hook is not None:
                self.frame_hook()

            now = time.monotonic()
            self.stats.sim_time = self.physics.time - t_sim_start
            self.stats.wall_time = now - t_wall_start
            if now >= t_status_next:
                _log.info("%s", self._summary())
                t_status_next = now + cfg.status_interval_s

            if cfg.max_sim_time is not None and self.physics.time >= cfg.max_sim_time:
                _log.info("reached max_sim_time=%.2f s, stopping", cfg.max_sim_time)
                break

            # Pacer: independent of PX4, so it also governs the boot window.
            # Unpaced, it still holds the fallback to real time.
            if frame_wall_dt is None and self._strict:
                t_wall_next = time.monotonic()
                continue
            t_wall_next += frame_wall_dt or cfg.imu_dt
            sleep_for = t_wall_next - time.monotonic()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            elif sleep_for < -1.0:
                # More than a second behind: we are CPU-bound, not ahead. Do not
                # try to catch up, or the pacer turns into a burst.
                t_wall_next = time.monotonic()

        self.running = False
        if self.controller is not None:
            self.controller.close()
        self.stats.sim_time = self.physics.time - t_sim_start
        self.stats.wall_time = time.monotonic() - t_wall_start
        _log.info("loop stopped: %s", self._summary())

    def _refuse(self, seq: int) -> None:
        self._refused_arm_cmds += 1
        if self._refused_arm_cmds <= 3 or self._refused_arm_cmds % 100 == 0:
            _log.warning(
                "side-channel arm_cmd seq %d refused (%d so far): an in-process "
                "controller drives the arm", seq, self._refused_arm_cmds,
            )

    def _summary(self) -> str:
        """The loop's health line, plus the arm's and the controller's."""
        parts = [self.stats.summary()]
        arm = self.physics.arm_status()
        if arm is not None:
            parts.append(arm.summary())
        if self.controller is not None:
            parts.append(self.controller.summary())
        return " ".join(parts)

    def stop(self) -> None:
        self.running = False
