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

# How long one message may sit unwritten before we start complaining. This is a
# warning threshold, not a deadline: see :meth:`HilServer.send`.
SEND_STALL_WARN_S = 1.0
# Poll granularity while waiting for writability.
_SEND_POLL_S = 0.1


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
        # Number of send() calls that had to wait for socket writability. A
        # non-zero value means PX4 stopped draining its receive buffer.
        self.send_stalls = 0
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
        """Send one packed message in full. Returns False if the peer went away.

        The socket is non-blocking, so a full send buffer surfaces as
        ``BlockingIOError`` rather than a wait. Two reasons this must be looped
        rather than passed to ``sendall``:

        * ``BlockingIOError`` is an ``OSError``, so catching it alongside the
          real disconnects would tear down a *live* PX4 link the moment it
          stopped draining its receive buffer for a moment.
        * ``sendall`` on a non-blocking socket may write part of the payload
          before raising, and abandoning a half-written MAVLink frame
          desynchronises PX4's parser silently -- exactly the failure plan 3.1
          warns about, and one that presents as a fault with no obvious cause.

        Once any byte of a message is out, the rest must follow, so this blocks
        until the whole payload is written. That is deliberate: it is real
        back-pressure from a PX4 that has stopped reading, and under lockstep a
        PX4 that never reads again is dead anyway -- its clock only advances on
        what we send here. A wall-clock watchdog is the right place to catch
        that (plan 7), not a partial write. We log every
        ``SEND_STALL_WARN_S`` so the stall is never silent.
        """
        if self._conn is None:
            return False
        view = memoryview(payload)
        sent = 0
        t_start = time.monotonic()
        t_warn_next = t_start + SEND_STALL_WARN_S
        stalled = False
        while sent < len(view):
            try:
                sent += self._conn.send(view[sent:])
                continue
            except (BlockingIOError, InterruptedError):
                stalled = True
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                _log.warning("send failed after %d/%d bytes: %s", sent, len(view), exc)
                self.drop_client()
                return False
            # Buffer full: wait for writability rather than dropping the client.
            select.select([], [self._conn], [], _SEND_POLL_S)
            now = time.monotonic()
            if now >= t_warn_next:
                t_warn_next = now + SEND_STALL_WARN_S
                _log.warning(
                    "PX4 has not drained its receive buffer for %.1f s "
                    "(%d/%d bytes of this message written); still waiting",
                    now - t_start, sent, len(view),
                )
        if stalled:
            self.send_stalls += 1
        # Advance the sequence counter, as pymavlink's own MAVLink.send() does
        # after packing. `msg.pack(link)` alone does not, and we pack and write
        # ourselves, so without this every frame we emit carries seq == 0. PX4
        # only feeds seq into its packet-loss statistics -- nothing misbehaves --
        # but a counter that never advances makes those statistics meaningless.
        self._mav.seq = (self._mav.seq + 1) % 256
        return True

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
