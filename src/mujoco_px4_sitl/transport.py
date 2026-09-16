"""TCP transport and MAVLink framing for the PX4 HIL link.

**We are the server.** ``px4-rc.mavlinksim`` starts ``simulator_mavlink -c
<port>``, which makes PX4 the TCP client retrying ``connect()`` every 500 us
(SimulatorMavlink.cpp:1154-1181). Binding early therefore lets us start before or
after PX4.
"""

from __future__ import annotations

import logging
import select
import socket
import time
from collections.abc import Iterator

from pymavlink.dialects.v20 import common as mavlink

_log = logging.getLogger(__name__)

# PX4 sends with MAV_SYS_ID / MAV_COMP_ID, default 1/1; we answer as a
# simulator-side pair that nothing in PX4 inspects.
OUR_SYSTEM_ID = 1
OUR_COMPONENT_ID = mavlink.MAV_COMP_ID_AUTOPILOT1


class HilServer:
    """One PX4 connection at a time, MAVLink v2 framed."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4560) -> None:
        self.host = host
        self.port = port
        self._listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen.bind((host, port))
        self._listen.listen(1)
        self._conn: socket.socket | None = None
        self._mav = mavlink.MAVLink(None, srcSystem=OUR_SYSTEM_ID, srcComponent=OUR_COMPONENT_ID)
        # Never raise on a partial or corrupt frame; PX4 also sends messages we
        # do not care about and must not desynchronise our parser (plan 3.1).
        self._mav.robust_parsing = True
        _log.info("listening for PX4 on tcp://%s:%d", host, port)

    # -- lifecycle ----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._conn is not None

    @property
    def mav(self) -> mavlink.MAVLink:
        """The framing object; pass it to the ``hil.encode_*`` functions."""
        return self._mav

    def accept(self, timeout: float | None = None) -> bool:
        """Wait for PX4 to connect. ``timeout`` is wall clock, seconds."""
        if self._conn is not None:
            return True
        self._listen.settimeout(timeout)
        try:
            conn, addr = self._listen.accept()
        except (TimeoutError, socket.timeout):
            return False
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setblocking(False)
        self._conn = conn
        _log.info("PX4 connected from %s:%d", *addr)
        return True

    def close(self) -> None:
        self.drop_client()
        try:
            self._listen.close()
        except OSError:
            pass

    def drop_client(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
            _log.warning("PX4 connection closed")

    # -- io -----------------------------------------------------------------

    def send(self, payload: bytes) -> bool:
        """Send a packed message. Returns False if the peer went away."""
        if self._conn is None:
            return False
        try:
            self._conn.sendall(payload)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            _log.warning("send failed: %s", exc)
            self.drop_client()
            return False

    def _recv_once(self) -> list[mavlink.MAVLink_message]:
        assert self._conn is not None
        try:
            data = self._conn.recv(4096)
        except (BlockingIOError, InterruptedError):
            return []
        except (ConnectionResetError, OSError) as exc:
            _log.warning("recv failed: %s", exc)
            self.drop_client()
            return []
        if not data:
            self.drop_client()
            return []
        return self._mav.parse_buffer(data) or []

    def drain(self) -> Iterator[mavlink.MAVLink_message]:
        """Yield every message currently readable, without blocking."""
        while self._conn is not None:
            readable, _, _ = select.select([self._conn], [], [], 0.0)
            if not readable:
                return
            messages = self._recv_once()
            if not messages:
                return
            yield from messages

    def wait(self, timeout: float) -> Iterator[mavlink.MAVLink_message]:
        """Block up to ``timeout`` **wall-clock** seconds for messages.

        Wall clock is mandatory here: a timeout expressed in simulated time never
        fires, because simulated time only advances when we send (plan 3.2).
        """
        deadline = time.monotonic() + timeout
        while self._conn is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            readable, _, _ = select.select([self._conn], [], [], remaining)
            if not readable:
                return
            messages = self._recv_once()
            if messages:
                yield from messages
                return
