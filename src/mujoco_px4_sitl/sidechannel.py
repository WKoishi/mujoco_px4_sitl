"""Simulator-private UDP side channel.

Our own schema, not MAVLink; PX4 never sees it (plan section 2). Carries full
ground truth outbound and arm joint commands inbound, so an external process --
typically a ROS 2 node in a *different* repository -- can drive the manipulator
and read state without touching this repo's internals.

JSON first, as the plan specifies: swap for a packed binary format only if
profiling says so. The ``v`` field is the schema version; bump it on any
incompatible change.
"""

from __future__ import annotations

import json
import logging
import select
import socket
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .sim import SimState

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_MAX_DATAGRAM = 65507


@dataclass
class ArmCommand:
    """Latest joint command received from the side channel."""

    seq: int = 0
    mode: str = "position"  # "position" | "torque"
    values: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))


class SideChannel:
    """Non-blocking UDP server. Never stalls the physics loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = 14650) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.subscribers: set[tuple[str, int]] = set()
        self.arm_command = ArmCommand()
        self._seq = 0
        _log.info("side channel on udp://%s:%d (schema v%d)", host, port, SCHEMA_VERSION)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    # -- inbound ------------------------------------------------------------

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
            values = msg.get("values", [])
            if not isinstance(values, list):
                _log.warning("side channel: arm_cmd.values must be a list")
                return
            self.arm_command = ArmCommand(
                seq=int(msg.get("seq", self.arm_command.seq + 1)),
                mode=str(msg.get("mode", "position")),
                values=np.asarray(values, dtype=np.float64),
            )
        else:
            _log.warning("side channel: unknown message type %r from %s:%d", kind, *addr)

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

    def serve(self, state: SimState, controls: NDArray[np.float64]) -> None:
        """Handle inbound datagrams, then publish ground truth. Call per tick."""
        self._poll()
        self._seq += 1
        self._publish({
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
        })
