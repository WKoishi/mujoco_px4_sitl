"""PX4's state estimate, read the way a companion computer reads it.

The in-process research controller (:mod:`control`) takes the vehicle's state
from here, not from MuJoCo: EKF2's ``vehicle_odometry`` as PX4 streams it in the
MAVLink ``ODOMETRY`` message on its API/offboard link (UDP 14540 + instance,
``px4-rc.mavlink``). Same message, same link, same path as a controller on the
real vehicle's companion computer.

**Which estimate has arrived is wall-clock; how old it is, is not.** PX4 schedules
the stream on its own clock, but the datagram reaches us whenever the socket
delivers it, so the newest estimate at a given simulated instant depends on wall
timing. Its age does not: ``ODOMETRY.time_usec`` is EKF2's IMU sample time on
PX4's clock, and under lockstep PX4's clock *is* ours -- ``hrt_absolute_time()``
returns the time our last ``HIL_SENSOR`` set (``drv_hrt.cpp:106-110``). So
``now - time`` is exact even when the arrival is not.

The positions are in EKF2's own local frame, whose datum it picks from GNSS; the
simulator's ground truth uses the configured home. :attr:`EstimateLink.origin`
carries EKF2's datum so the two can be compared (:func:`frames.rebase_ned`).
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pymavlink.dialects.v20 import common as _MAV

from .frames import GeodeticProjection

_log = logging.getLogger(__name__)

# A companion computer's identity. Only the component id means anything to PX4
# here; the system id just has to differ from the vehicle's.
SOURCE_SYSTEM = 245
SOURCE_COMPONENT = _MAV.MAV_COMP_ID_ONBOARD_COMPUTER
_MAX_DATAGRAM = 65535
# How long to wait for an ACK before asking for the stream rate again.
_REQUEST_RETRY_S = 1.0


@dataclass(frozen=True)
class Px4Estimate:
    """One ``ODOMETRY`` from PX4: EKF2's estimate, in PX4 frames and SI units."""

    time: float  # EKF2's IMU sample time, simulated seconds
    received: float  # simulated time at which it was first read here
    pos_ned: NDArray[np.float64]  # EKF2's local frame; see EstimateLink.origin
    vel_ned: NDArray[np.float64]
    q_frd_ned: NDArray[np.float64]  # body FRD -> local NED, [w, x, y, z]
    rates_frd: NDArray[np.float64]  # bias-corrected, body FRD
    # Bumps on every EKF2 state reset: a step in the estimate that is not motion.
    reset_counter: int
    quality: int  # 0 unknown, else 1 (worst) .. 100 (best)


class EstimateLink:
    """Non-blocking reader of PX4's ``ODOMETRY`` stream on the API link.

    Asks PX4 for the stream at ``rate_hz`` (the onboard-mode default is 30 Hz,
    ``mavlink_main.cpp``) once PX4's heartbeat reveals where to send the request,
    and keeps asking until PX4 acknowledges it.
    """

    def __init__(self, host: str, port: int, rate_hz: float) -> None:
        if rate_hz <= 0.0:
            raise ValueError("estimate rate must be > 0 Hz")
        self.rate_hz = float(rate_hz)
        # A plain socket and the v2 dialect, not mavutil: mavutil starts in
        # MAVLink 1 and switches the whole process's dialect when it first sees a
        # v2 frame, and ODOMETRY only exists in v2.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((host, port))
        except OSError as exc:
            self.sock.close()
            raise OSError(
                f"cannot bind udp://{host}:{port} for PX4's API link ({exc}). "
                f"Another MAVLink client (MAVSDK, MAVROS) may hold it"
            ) from exc
        self.sock.setblocking(False)
        self.mav = _MAV.MAVLink(None, srcSystem=SOURCE_SYSTEM, srcComponent=SOURCE_COMPONENT)
        self.mav.robust_parsing = True
        self._px4_addr: tuple[str, int] | None = None
        self.latest: Px4Estimate | None = None
        self.origin: GeodeticProjection | None = None
        self.received = 0
        self.out_of_order = 0
        self.rejected = 0
        # Simulated seconds from EKF2's sample to our first read of it: the part
        # of an estimate's age that wall clock decides.
        self.max_delivery = 0.0
        self._target: tuple[int, int] | None = None
        self._rate_acked = False
        self._last_request = -np.inf
        _log.info("PX4 estimate: ODOMETRY on udp://%s:%d, asking for %.0f Hz",
                  host, port, rate_hz)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def poll(self, now: float) -> Px4Estimate | None:
        """Read everything pending; return the newest estimate so far.

        ``now`` is the simulated time, stamped on estimates first seen here.
        """
        while True:
            try:
                data, addr = self.sock.recvfrom(_MAX_DATAGRAM)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as exc:
                # ECONNREFUSED from an earlier send, for one: not fatal here.
                _log.debug("API link: %s", exc)
                break
            self._px4_addr = addr
            for msg in self.mav.parse_buffer(data) or []:
                self._handle(msg, now)
        self._request_rate()
        return self.latest

    def _handle(self, msg, now: float) -> None:
        kind = msg.get_type()
        if kind == "ODOMETRY":
            self._odometry(msg, now)
        elif kind == "HEARTBEAT":
            if self._target is None and msg.autopilot != _MAV.MAV_AUTOPILOT_INVALID:
                self._target = (msg.get_srcSystem(), msg.get_srcComponent())
        elif kind == "GPS_GLOBAL_ORIGIN":
            origin = (msg.latitude * 1e-7, msg.longitude * 1e-7, msg.altitude * 1e-3)
            if self.origin is None or origin != (
                self.origin.ref_lat_deg, self.origin.ref_lon_deg, self.origin.ref_alt_m
            ):
                if self.origin is not None:
                    _log.warning("EKF2 moved its local origin: estimates before "
                                 "and after are in different frames")
                self.origin = GeodeticProjection(*origin)
        elif kind == "COMMAND_ACK" and msg.command == _MAV.MAV_CMD_SET_MESSAGE_INTERVAL:
            if msg.result == _MAV.MAV_RESULT_ACCEPTED:
                if not self._rate_acked:
                    _log.info("PX4 accepted ODOMETRY at %.0f Hz", self.rate_hz)
                self._rate_acked = True
            elif not self._rate_acked:
                _log.warning("PX4 refused the ODOMETRY rate request (result %d); "
                             "retrying", msg.result)

    def _odometry(self, msg, now: float) -> None:
        # Pose and velocity must both be local NED, which is what EKF2 publishes
        # (EKF2.cpp PublishOdometry). Anything else needs a rotation, and those
        # live only in frames.py -- refuse rather than rotate here.
        ned = _MAV.MAV_FRAME_LOCAL_NED
        if msg.frame_id != ned or msg.child_frame_id != ned:
            self.rejected += 1
            if self.rejected <= 3 or self.rejected % 100 == 0:
                _log.warning(
                    "ODOMETRY in frames %d/%d, not local NED/NED; ignored (%d so far)",
                    msg.frame_id, msg.child_frame_id, self.rejected,
                )
            return
        sample = msg.time_usec * 1e-6
        if self.latest is not None and sample <= self.latest.time:
            self.out_of_order += 1  # UDP reordering, or a duplicate
            return
        self.received += 1
        self.max_delivery = max(self.max_delivery, now - sample)
        self.latest = Px4Estimate(
            time=sample,
            received=now,
            pos_ned=np.array([msg.x, msg.y, msg.z], dtype=np.float64),
            vel_ned=np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64),
            q_frd_ned=np.array(msg.q, dtype=np.float64),
            rates_frd=np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed],
                               dtype=np.float64),
            reset_counter=int(msg.reset_counter),
            quality=int(getattr(msg, "quality", 0)),
        )

    def _request_rate(self) -> None:
        # Wall clock on purpose: this is a link handshake, not part of the loop's
        # timing, and it should retry even while simulated time is held.
        if self._rate_acked or self._target is None or self._px4_addr is None:
            return
        wall = time.monotonic()
        if wall - self._last_request < _REQUEST_RETRY_S:
            return
        self._last_request = wall
        request = self.mav.command_long_encode(
            self._target[0], self._target[1], _MAV.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            float(_MAV.MAVLINK_MSG_ID_ODOMETRY), 1e6 / self.rate_hz, 0, 0, 0, 0, 0,
        )
        try:
            self.sock.sendto(request.pack(self.mav), self._px4_addr)
        except OSError as exc:
            _log.debug("API link: rate request not sent (%s)", exc)
