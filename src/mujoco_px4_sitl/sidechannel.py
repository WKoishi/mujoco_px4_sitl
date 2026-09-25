"""Simulator-private UDP side channel.

Our own schema, not MAVLink; PX4 never sees it (plan section 2). Carries full
ground truth outbound and arm joint commands inbound, so an external process --
typically a ROS 2 node in a *different* repository -- can drive the manipulator
and read state without touching this repo's internals.

JSON first, as the plan specifies: swap for a packed binary format only if
profiling says so. The ``v`` field is the schema version; bump it on any
incompatible change. Optional fields added since v1 -- ``arm_cmd.state_time``
and ``ground_truth.arm`` -- are ignorable by an older peer, so they did not.

Inbound is polled every frame and outbound published at ``sidechannel_rate_hz``.
The asymmetry is deliberate: polling at the publish rate would hold an arriving
``arm_cmd`` for up to one publish period, a delay no controller asked for and
the research could not tell from its own.
"""

from __future__ import annotations

import json
import logging
import select
import socket

import numpy as np
from numpy.typing import NDArray

from .arm import ArmCommand, ArmStatus
from .sim import SimState

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_MAX_DATAGRAM = 65507


class SideChannel:
    """Non-blocking UDP server. Never stalls the physics loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = 14650) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.subscribers: set[tuple[str, int]] = set()
        self._arm_commands: list[ArmCommand] = []
        self._arm_seq = 0
        self._arm_malformed = 0
        self._seq = 0
        _log.info("side channel on udp://%s:%d (schema v%d)", host, port, SCHEMA_VERSION)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    # -- inbound ------------------------------------------------------------

    def poll(self) -> list[ArmCommand]:
        """Handle every pending datagram; return the ``arm_cmd``s in arrival
        order. All of them, not only the newest: ordering by ``state_time`` is
        the servos' decision, and they can only make it if they see each one."""
        self._arm_commands = []
        self._poll()
        return self._arm_commands

    def _poll(self) -> None:
        while True:
            readable, _, _ = select.select([self.sock], [], [], 0.0)
            if not readable:
                return
            try:
                data, addr = self.sock.recvfrom(_MAX_DATAGRAM)
            except (BlockingIOError, OSError):
                return
            self._handle(data, addr)

    def _handle(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _log.warning("side channel: undecodable datagram from %s:%d (%s)", *addr, exc)
            return
        if not isinstance(msg, dict):
            _log.warning("side channel: expected a JSON object from %s:%d", *addr)
            return
        version = msg.get("v")
        if version != SCHEMA_VERSION:
            _log.warning("side channel: schema v%s from %s:%d, we speak v%d",
                         version, *addr, SCHEMA_VERSION)
            return

        kind = msg.get("type")
        if kind == "subscribe":
            self.subscribers.add(addr)
            _log.info("side channel: %s:%d subscribed", *addr)
        elif kind == "unsubscribe":
            self.subscribers.discard(addr)
            _log.info("side channel: %s:%d unsubscribed", *addr)
        elif kind == "arm_cmd":
            command = self._parse_arm_cmd(msg, addr)
            if command is not None:
                self._arm_commands.append(command)
        else:
            _log.warning("side channel: unknown message type %r from %s:%d", kind, *addr)

    def _parse_arm_cmd(self, msg: dict, addr: tuple[str, int]) -> ArmCommand | None:
        """Type-check an ``arm_cmd``. Returns None, and warns, if it is malformed.

        Every conversion is guarded: this runs inside the lockstep loop, and an
        exception here -- a string among the values was enough -- would take the
        simulator, and PX4's clock with it, down.
        """
        values = msg.get("values", [])
        state_time = msg.get("state_time")
        mode = msg.get("mode", "position")
        try:
            if not isinstance(values, list):
                raise ValueError("values must be a list")
            if not isinstance(mode, str):
                raise ValueError("mode must be a string")
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError("values must be a flat list of numbers")
            seq = int(msg.get("seq", self._arm_seq + 1))
            state_time = None if state_time is None else float(state_time)
        except (TypeError, ValueError) as exc:
            self._arm_malformed += 1
            if self._arm_malformed <= 3 or self._arm_malformed % 100 == 0:
                _log.warning("side channel: malformed arm_cmd from %s:%d (%s; %d so far)",
                             *addr, exc, self._arm_malformed)
            return None
        self._arm_seq = seq
        return ArmCommand(seq=seq, mode=mode, values=array, state_time=state_time)

    # -- outbound -----------------------------------------------------------

    def _publish(self, payload: dict) -> None:
        if not self.subscribers:
            return
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for addr in list(self.subscribers):
            try:
                self.sock.sendto(blob, addr)
            except OSError as exc:
                _log.warning("side channel: dropping %s:%d (%s)", *addr, exc)
                self.subscribers.discard(addr)

    def publish(
        self, state: SimState, controls: NDArray[np.float64], arm: ArmStatus | None = None
    ) -> None:
        """Publish one ``ground_truth``. ``arm`` adds the ``arm`` block."""
        self._seq += 1
        payload = {
            "v": SCHEMA_VERSION,
            "type": "ground_truth",
            "seq": self._seq,
            "time": round(state.time, 6),
            # PX4 frames throughout, so the two channels never disagree.
            "frame": "NED/FRD",
            "pos_ned": [round(float(v), 6) for v in state.pos_ned],
            "vel_ned": [round(float(v), 6) for v in state.vel_ned],
            "q_frd_ned": [round(float(v), 9) for v in state.q_px4],
            "rates_frd": [round(float(v), 6) for v in state.gyro_frd],
            "accel_frd": [round(float(v), 6) for v in state.accel_frd],
            "geodetic": [state.lat_deg, state.lon_deg, round(state.alt_m, 4)],
            "actuators": [round(float(v), 6) for v in controls],
        }
        if arm is not None:
            # Command bookkeeping and clearance, not joint state: whether an
            # out-of-process controller should get qpos is the topology question
            # of AGENTS.md section 3, not a field to add here.
            payload["arm"] = {
                "cmd_seq": arm.cmd_seq,
                "cmd_age": None if arm.cmd_age is None else round(arm.cmd_age, 6),
                "cmd_stale": arm.cmd_stale,
                "prop_clearance": (
                    None if arm.prop_clearance is None else round(arm.prop_clearance, 6)
                ),
            }
        self._publish(payload)
