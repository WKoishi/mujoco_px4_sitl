"""PX4's API link, used the way a companion computer uses it, on simulated time.

The in-process research controller (:mod:`control`) takes the vehicle's state
from here, not from MuJoCo: EKF2's ``vehicle_odometry`` in the MAVLink
``ODOMETRY`` message on PX4's API/offboard link (UDP 14540 + instance,
``px4-rc.mavlink``). Its setpoints and commands go back the same way. Same
messages, same link, as a controller on the real vehicle's companion computer.

**What makes it deterministic is PX4's receive thread** (plan 3.9). It runs on
wall clock, handles a datagram's messages in order and each to completion, and
answers ``PING`` from the same thread. So a ``PING`` at the end of a datagram,
once echoed, proves everything before it is in uORB -- a **barrier** that works
while PX4's clock is held between our frames. :meth:`ApiLink.send` uses one, and
:meth:`ApiLink.fetch` asks for EKF2's newest output behind one, after every
frame PX4 has answered.

Two things would break that, and are kept off the barrier path:

* **Requests that end in ``configure_stream_threadsafe()``** block the receive
  thread on simulated time, so a barrier behind one never echoes while the clock
  is held. :func:`check_barrier_safe` refuses them. The one this link needs,
  silencing the periodic ``ODOMETRY`` stream, is sent before the first barrier,
  while frames advance without barriers, between two PINGs: barriers start
  once PX4 has acknowledged it and echoed the second, which shows the request
  has left the receive thread. The first makes PX4 register our component
  before handling the request; PX4 sends a ``COMMAND_ACK`` only to a component
  it has seen, and registers one only after handling its message
  (``mavlink_main.cpp:2692``, ``mavlink_receiver.cpp:3244-3249``), so an ACK to
  the very first message can be dropped.
* **The periodic stream races the estimator**: it sends whatever is newest when
  PX4's main thread wakes. Given an interval of 2e9 us it never sends on its
  own, and ``REQUEST_MESSAGE`` then sends only a new output. Not -1: that
  deletes the stream, and the next request would re-create it through the
  blocking path.

**Ages are exact.** ``ODOMETRY.time_usec`` is EKF2's IMU sample time on PX4's
clock, and under lockstep PX4's clock *is* ours -- ``hrt_absolute_time()``
returns the time our last ``HIL_SENSOR`` set (``drv_hrt.cpp:106-110``).

The positions are in EKF2's own local frame, whose datum it picks from GNSS; the
simulator's ground truth uses the configured home. :attr:`ApiLink.origin`
carries EKF2's datum so the two can be compared (:func:`frames.rebase_ned`).
"""

from __future__ import annotations

import bisect
import logging
import math
import select
import socket
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymavlink.dialects.v20 import common as _MAV

from .frames import GeodeticProjection

_log = logging.getLogger(__name__)

# A companion computer's identity. Only the component id means anything to PX4
# here; the system id just has to differ from the vehicle's.
SOURCE_SYSTEM = 245
SOURCE_COMPONENT = _MAV.MAV_COMP_ID_ONBOARD_COMPUTER
_MAX_DATAGRAM = 65535
# PX4 reads 8000 bytes per datagram on posix (mavlink_receiver.cpp:3137); stay
# well under, splitting a longer send across datagrams, which the loopback
# delivers and PX4 handles in order.
_DATAGRAM_BUDGET = 1400
# The silencing handshake, wall clock on purpose: a link handshake, not part of
# the loop's timing. A request PX4 has handled but not accepted is asked again
# after the first; one whose closing PING never echoes was lost on the way, and
# is asked again after the second.
_REQUEST_RETRY_S = 1.0
_REQUEST_LOST_S = 5.0
# An interval no run reaches, in us: the stream stays, and never sends.
_SILENT_INTERVAL_US = 2e9
# Estimates kept for lookups by sample time: 1 s at EKF2's 125 Hz.
_HISTORY = 128

# Commands whose handling ends in configure_stream_threadsafe(), which sleeps
# on simulated time until PX4's main thread takes the request
# (mavlink_main.cpp:1207-1230): SET_MESSAGE_INTERVAL (mavlink_receiver.cpp:2285)
# and every path into handle_request_message_command (:569-601, :762-790), which
# does so for a stream the link does not have yet. Which streams it has cannot be
# seen from here, so all of them are refused.
_BLOCKING_COMMANDS = frozenset({
    _MAV.MAV_CMD_SET_MESSAGE_INTERVAL,
    _MAV.MAV_CMD_REQUEST_MESSAGE,
    _MAV.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
    _MAV.MAV_CMD_REQUEST_PROTOCOL_VERSION,
    _MAV.MAV_CMD_GET_HOME_POSITION,
    _MAV.MAV_CMD_REQUEST_FLIGHT_INFORMATION,
    _MAV.MAV_CMD_REQUEST_STORAGE_INFORMATION,
})
# SET_GPS_GLOBAL_ORIGIN ends by requesting GPS_GLOBAL_ORIGIN (:1292).
_BLOCKING_MESSAGES = frozenset({"SET_GPS_GLOBAL_ORIGIN"})


def check_barrier_safe(msg: _MAV.MAVLink_message) -> None:
    """Raise if handling ``msg`` can block PX4's receive thread on simulated
    time, which would hold every barrier behind it until the clock moves."""
    kind = msg.get_type()
    if kind in _BLOCKING_MESSAGES or (
        kind in ("COMMAND_LONG", "COMMAND_INT") and msg.command in _BLOCKING_COMMANDS
    ):
        what = kind if kind in _BLOCKING_MESSAGES else f"{kind} {msg.command}"
        raise ValueError(
            f"{what} ends in configure_stream_threadsafe(), which blocks PX4's "
            f"receive thread on simulated time: refused on the barrier path "
            f"(IMPLEMENTATION_PLAN.md 3.9)"
        )


# --- builders for what a controller sends ----------------------------------
# Target ids are left 0; ApiLink.send fills in PX4's from its heartbeat.


def command(cmd: int, *params: float) -> _MAV.MAVLink_command_long_message:
    """``COMMAND_LONG``, up to seven params, the rest 0."""
    if len(params) > 7:
        raise ValueError("COMMAND_LONG has seven params")
    p = [float(v) for v in params] + [0.0] * (7 - len(params))
    return _MAV.MAVLink_command_long_message(0, 0, int(cmd), 0, *p)


def position_setpoint(
    pos_ned: ArrayLike, yaw: float | None = None
) -> _MAV.MAVLink_set_position_target_local_ned_message:
    """Position in EKF2's local NED frame, m; yaw in rad if given."""
    n, e, d = (float(v) for v in np.asarray(pos_ned, dtype=np.float64))
    # Ignore velocity, acceleration, force and yaw rate; yaw too unless given.
    mask = 0b0000_1011_1111_1000 if yaw is not None else 0b0000_1111_1111_1000
    return _MAV.MAVLink_set_position_target_local_ned_message(
        0, 0, 0, _MAV.MAV_FRAME_LOCAL_NED, mask, n, e, d,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 if yaw is None else float(yaw), 0.0,
    )


def body_rate_setpoint(
    rates_frd: ArrayLike, thrust: float
) -> _MAV.MAVLink_set_attitude_target_message:
    """Body rates, rad/s FRD, and collective thrust, normalized ``[0, 1]``."""
    p, q, r = (float(v) for v in np.asarray(rates_frd, dtype=np.float64))
    return _MAV.MAVLink_set_attitude_target_message(
        0, 0, 0, _MAV.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
        [1.0, 0.0, 0.0, 0.0], p, q, r, float(thrust),
    )


def attitude_setpoint(
    q_frd_ned: ArrayLike, thrust: float
) -> _MAV.MAVLink_set_attitude_target_message:
    """Attitude, body FRD -> NED ``[w, x, y, z]``, and collective thrust."""
    q = [float(v) for v in np.asarray(q_frd_ned, dtype=np.float64)]
    rates_ignored = (
        _MAV.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
        | _MAV.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
        | _MAV.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )
    return _MAV.MAVLink_set_attitude_target_message(
        0, 0, 0, rates_ignored, q, 0.0, 0.0, 0.0, float(thrust),
    )


@dataclass(frozen=True)
class Px4Estimate:
    """One ``ODOMETRY`` from PX4: EKF2's estimate, in PX4 frames and SI units."""

    time: float  # EKF2's IMU sample time, simulated seconds
    received: float  # simulated time at which it was first read here
    pos_ned: NDArray[np.float64]  # EKF2's local frame; see ApiLink.origin
    vel_ned: NDArray[np.float64]
    q_frd_ned: NDArray[np.float64]  # body FRD -> local NED, [w, x, y, z]
    rates_frd: NDArray[np.float64]  # bias-corrected, body FRD
    # Bumps on every EKF2 state reset: a step in the estimate that is not motion.
    reset_counter: int
    quality: int  # 0 unknown, else 1 (worst) .. 100 (best)


@dataclass(frozen=True)
class CommandAck:
    """A ``COMMAND_ACK`` PX4 sent to us."""

    command: int
    result: int  # MAV_RESULT
    received: float  # simulated time at which it was read here


class ApiLink:
    """PX4's API link: estimates in, setpoints and commands out.

    Everything is non-blocking except the barriers, which wait on wall clock for
    at most ``barrier_timeout`` each. A barrier that times out is logged and
    counted, and later sends go unbarriered -- still with a ``PING`` at the end,
    so its echo shows when PX4's receive thread is back.
    """

    def __init__(self, host: str, port: int, barrier_timeout: float = 0.2) -> None:
        if barrier_timeout <= 0.0:
            raise ValueError("barrier timeout must be > 0 (wall clock)")
        self.barrier_timeout = float(barrier_timeout)
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
        self._target: tuple[int, int] | None = None

        self.latest: Px4Estimate | None = None
        self._history: deque[Px4Estimate] = deque(maxlen=_HISTORY)
        self._times: deque[float] = deque(maxlen=_HISTORY)
        self.origin: GeodeticProjection | None = None
        self.acks: list[CommandAck] = []
        self.received = 0
        self.out_of_order = 0
        self.rejected = 0

        # Silencing the periodic ODOMETRY, one request at a time: it is in
        # PX4's receive thread until the PING sent after it echoes.
        self._silenced = False
        self._silence_ping: int | None = None
        self._last_request = -math.inf
        # PING sequence numbers: the last sent, and the newest echoed.
        self._ping_seq = 0
        self._echoed = 0
        self._barrier_ok = True
        self.barriers = 0
        self.barrier_timeouts = 0
        self.unbarriered = 0
        self.fetches = 0
        _log.info("PX4 API link on udp://%s:%d", host, port)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    # -- state ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Barriers are usable: PX4 has been heard, the periodic ODOMETRY is
        silenced, and no request that blocks the receive thread is in flight."""
        return self._silenced and self._echoed >= (self._silence_ping or 0)

    def estimate_at(self, t: float) -> Px4Estimate | None:
        """The newest estimate received with sample time at or before ``t``."""
        i = bisect.bisect_right(self._times, t + 1e-9)
        return self._history[i - 1] if i else None

    # -- inbound ----------------------------------------------------------

    def poll(self, now: float) -> Px4Estimate | None:
        """Read everything pending; return the newest estimate so far.

        ``now`` is the simulated time, stamped on what is first seen here. Also
        drives the silencing handshake.
        """
        self._read(now)
        self._silence()
        return self.latest

    def _read(self, now: float) -> None:
        while True:
            try:
                data, addr = self.sock.recvfrom(_MAX_DATAGRAM)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as exc:
                # ECONNREFUSED from an earlier send, for one: not fatal here.
                _log.debug("API link: %s", exc)
                return
            self._px4_addr = addr
            for msg in self.mav.parse_buffer(data) or []:
                self._handle(msg, now)

    def _handle(self, msg, now: float) -> None:
        kind = msg.get_type()
        if kind == "ODOMETRY":
            self._odometry(msg, now)
        elif kind == "PING":
            if (msg.target_system, msg.target_component) == (SOURCE_SYSTEM, SOURCE_COMPONENT):
                self._echoed = max(self._echoed, int(msg.seq))
                if not self._barrier_ok and self._echoed == self._ping_seq:
                    self._barrier_ok = True
                    _log.warning("PX4's receive thread echoes again: barriers resume")
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
        elif kind == "COMMAND_ACK":
            self._ack(msg, now)

    def _ack(self, msg, now: float) -> None:
        if msg.command == _MAV.MAV_CMD_SET_MESSAGE_INTERVAL and self._silence_ping is not None:
            if msg.result == _MAV.MAV_RESULT_ACCEPTED:
                self._silenced = True
                _log.info("PX4 silenced the periodic ODOMETRY: barriers from here")
            else:
                _log.warning("PX4 refused to silence ODOMETRY (result %d); retrying",
                             msg.result)
            return
        if msg.command == _MAV.MAV_CMD_REQUEST_MESSAGE:
            return  # one per fetch; DENIED just means no new output
        self.acks.append(CommandAck(int(msg.command), int(msg.result), now))

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
        self._history.append(self.latest)
        self._times.append(sample)

    # -- outbound ---------------------------------------------------------

    def _silence(self) -> None:
        """Ask for the periodic ODOMETRY at an interval no run reaches, until
        PX4 has acknowledged it. Sent without a barrier: the request blocks the
        receive thread until PX4's clock moves, which it does between frames."""
        if self._silenced or self._target is None or self._px4_addr is None:
            return
        wall = time.monotonic()
        if self._silence_ping is not None:
            if self._echoed < self._silence_ping:
                if wall - self._last_request < _REQUEST_LOST_S:
                    return
                _log.warning("silencing ODOMETRY: no echo in %.0f s; asking again",
                             _REQUEST_LOST_S)
            elif wall - self._last_request < _REQUEST_RETRY_S:
                return
        request = command(_MAV.MAV_CMD_SET_MESSAGE_INTERVAL,
                          _MAV.MAVLINK_MSG_ID_ODOMETRY, _SILENT_INTERVAL_US)
        if self._transmit([self._ping(), request, self._ping()]):
            self._silence_ping = self._ping_seq
            self._last_request = wall

    def _ping(self) -> _MAV.MAVLink_ping_message:
        """A PING request, targets 0/0, with the next sequence number."""
        self._ping_seq += 1
        return _MAV.MAVLink_ping_message(0, self._ping_seq, 0, 0)

    def _pack(self, msg: _MAV.MAVLink_message) -> bytes:
        # A PING with targets 0/0 is a request, which PX4 echoes; with its own
        # ids it would be taken for an echo of PX4's, and answered by nothing.
        if msg.get_type() != "PING":
            for field, value in zip(("target_system", "target_component"),
                                    self._target or (0, 0)):
                if getattr(msg, field, None) == 0:
                    setattr(msg, field, value)
        data = msg.pack(self.mav)
        # pack() alone does not advance the sequence number; MAVLink.send() would.
        self.mav.seq = (self.mav.seq + 1) % 256
        return data

    def _transmit(self, messages: Sequence[_MAV.MAVLink_message]) -> bool:
        if self._px4_addr is None:
            return False
        datagrams: list[bytes] = [b""]
        for msg in messages:
            data = self._pack(msg)
            if datagrams[-1] and len(datagrams[-1]) + len(data) > _DATAGRAM_BUDGET:
                datagrams.append(b"")
            datagrams[-1] += data
        try:
            for datagram in datagrams:
                self.sock.sendto(datagram, self._px4_addr)
        except OSError as exc:
            _log.warning("API link: send failed (%s)", exc)
            return False
        return True

    def send(self, messages: Sequence[_MAV.MAVLink_message], now: float) -> bool:
        """Send ``messages`` in order, behind a barrier when the link is ready.

        True means PX4 echoed the barrier: every message is in uORB, or refused
        by PX4, before this returns. False means they went out unproven -- the
        link is not ready, or the echo did not come within the timeout. Raises
        if a message could block PX4's receive thread (:func:`check_barrier_safe`).
        """
        for msg in messages:
            check_barrier_safe(msg)
        return self._send(list(messages), now)

    def fetch(self, now: float) -> bool:
        """Ask for EKF2's newest output behind a barrier; True if proven.

        Sent after every frame PX4 has answered, not only at samples: the stream
        sends only the newest output, and the one a sample needs is older.
        """
        if not self.ready:
            return False
        self.fetches += 1
        request = command(_MAV.MAV_CMD_REQUEST_MESSAGE, _MAV.MAVLINK_MSG_ID_ODOMETRY)
        return self._send([request], now)

    def _send(self, messages: list[_MAV.MAVLink_message], now: float) -> bool:
        self._read(now)
        if not self.ready:
            if messages and self._transmit(messages):
                self.unbarriered += 1
            return False
        if not self._transmit(messages + [self._ping()]):
            return False
        if not self._barrier_ok:
            self.unbarriered += 1
            return False
        deadline = time.monotonic() + self.barrier_timeout
        while self._echoed < self._ping_seq:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.barrier_timeouts += 1
                self._barrier_ok = False
                _log.warning(
                    "PING barrier not echoed within %.0f ms wall clock (%d so far): "
                    "sends go unbarriered until PX4's receive thread answers",
                    self.barrier_timeout * 1e3, self.barrier_timeouts,
                )
                return False
            select.select([self.sock], [], [], remaining)
            self._read(now)
        self.barriers += 1
        return True

    def summary(self) -> str:
        return (
            f"api barriers={self.barriers} barrier_timeouts={self.barrier_timeouts} "
            f"unbarriered={self.unbarriered}"
        )
