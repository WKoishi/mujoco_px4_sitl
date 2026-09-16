"""Lockstep orchestration: the plan section 3.2 loop, complete.

One regime, not two. **Wall clock paces the loop; the actuator stream only bounds
how far ahead of PX4 we may get.** Both halves are mandatory:

* Without the pacer, nothing bounds us during PX4's boot -- which happens
  entirely inside this loop's first frames, because ``simulator_mavlink start``
  blocks ``rcS`` until our first ``HIL_SENSOR``. Running ahead drops IMU FIFO
  samples and presents as estimator divergence.
* Without the brake, we can outrun PX4's pipeline in flight the same way.
* Blocking on ``HIL_ACTUATOR_CONTROLS`` *per frame* deadlocks at startup:
  PX4 publishes none before Commander is up, and its escape hatches are either
  measured in simulated time or compiled out under lockstep.

The brake's timeout is therefore **wall clock**, and its expiry is never fatal.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pymavlink.dialects.v20 import common as mavlink

from . import hil
from .config import Config
from .sim import Physics
from .sidechannel import SideChannel
from .transport import HilServer

_log = logging.getLogger(__name__)


@dataclass
class LoopStats:
    """Counters that make the two section 7 loop faults visible."""

    frames: int = 0
    actuator_messages: int = 0
    brake_waits: int = 0
    brake_timeouts: int = 0
    discarded_messages: int = 0
    # Sim/wall time spent *inside* the loop, excluding connection setup. The
    # ratio of the two is the section 7 diagnostic: it should sit near
    # speed_factor, and neither grow without bound (we outran PX4, IMU FIFO
    # samples dropped) nor collapse toward zero (the brake is pacing the loop).
    sim_time: float = 0.0
    wall_time: float = 0.0

    @property
    def ratio(self) -> float:
        return (self.sim_time / self.wall_time) if self.wall_time > 0.0 else 0.0

    def summary(self) -> str:
        return (
            f"t_sim={self.sim_time:8.2f}s ratio={self.ratio:5.3f} "
            f"frames={self.frames} act={self.actuator_messages} "
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
    ) -> None:
        self.cfg = cfg
        self.physics = physics
        self.server = server
        self.sidechannel = sidechannel
        # Called once per IMU frame. Used for the viewer, which must never gate
        # the physics loop, so it decimates internally.
        self.frame_hook = frame_hook
        self.stats = LoopStats()
        self.controls = hil.ActuatorControls()
        self.running = False

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

        frame_wall_dt = cfg.imu_dt / cfg.speed_factor
        t_wall_start = time.monotonic()
        t_sim_start = self.physics.time
        t_wall_next = t_wall_start
        t_status_next = t_wall_start + cfg.status_interval_s
        controls = np.zeros(self.physics.num_actuators)

        _log.info(
            "loop start: IMU %.0f Hz, speed x%.2f, max_lead=%d frames, "
            "brake timeout %.0f ms",
            cfg.imu_rate_hz, cfg.speed_factor, cfg.max_lead_frames,
            cfg.brake_timeout_s * 1e3,
        )

        while self.running:
            if not self._send_frame():
                _log.warning("PX4 link lost, stopping loop")
                break
            self._frames_since_ack += 1

            self._drain()
            if self._frames_since_ack >= cfg.max_lead_frames:
                self._brake()

            controls = self.controls.effective(self.physics.num_actuators)
            self.physics.step_frame(controls)
            self.stats.frames += 1

            if self.sidechannel is not None and self.stats.frames % self._sidechannel_decimation == 0:
                self.sidechannel.serve(self.physics.state(), controls)

            if self.frame_hook is not None:
                self.frame_hook()

            now = time.monotonic()
            self.stats.sim_time = self.physics.time - t_sim_start
            self.stats.wall_time = now - t_wall_start
            if now >= t_status_next:
                _log.info("%s", self.stats.summary())
                t_status_next = now + cfg.status_interval_s

            if cfg.max_sim_time is not None and self.physics.time >= cfg.max_sim_time:
                _log.info("reached max_sim_time=%.2f s, stopping", cfg.max_sim_time)
                break

            # Pacer: independent of PX4, so it also governs the boot window.
            t_wall_next += frame_wall_dt
            sleep_for = t_wall_next - time.monotonic()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            elif sleep_for < -1.0:
                # More than a second behind: we are CPU-bound, not ahead. Do not
                # try to catch up, or the pacer turns into a burst.
                t_wall_next = time.monotonic()

        self.running = False
        self.stats.sim_time = self.physics.time - t_sim_start
        self.stats.wall_time = time.monotonic() - t_wall_start
        _log.info("loop stopped: %s", self.stats.summary())

    def stop(self) -> None:
        self.running = False
