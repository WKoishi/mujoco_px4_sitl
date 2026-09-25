#!/usr/bin/env python3
"""Minimal side-channel client: the reference for the UDP schema.

Deliberately standalone. It imports nothing from this repository, which is what
phase 6 requires of an external consumer. A ROS 2 bridge node in a separate
package should look like this, with rclpy publishing whatever it reads here.

    python scripts/sidechannel_example.py --count 5
    python scripts/sidechannel_example.py --latest --count 5
    python scripts/sidechannel_example.py --arm-cmd 0.5,0,0,0,0 --count 100

**A consumer slower than the publish rate must drain, or it reads stale state.**
Ground truth is published at 50 Hz into a socket buffer, so a client that polls
once a second gets a datagram from ~a second ago, and one that polls after a long
sleep gets a datagram from the *start* of that sleep. `--latest` shows the drain
pattern: read until the socket is empty and keep the newest. Any bridge node that
samples on its own timer needs this.

**An arm command is a stream, not a request.** The simulator's arm watchdog
(``--arm-timeout``, 0.5 s of simulated time by default) treats a command older
than that as a controller that has stopped, and acts on it (``--arm-on-timeout``).
So ``--arm-cmd`` resends the command with every ground_truth it reads, echoing
that message's ``time`` as ``state_time``: the command's age is then the age of
the state it was computed from, which ground_truth reports back as
``arm.cmd_age``. With ``--latest`` that is one command per poll, so an
``--interval`` longer than the timeout shows the watchdog tripping.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

SCHEMA_VERSION = 1


def _fmt(values: list[float]) -> str:
    return "[" + " ".join(f"{v:+7.3f}" for v in values) + "]"


def drain_to_latest(sock: socket.socket) -> dict | None:
    """Discard the backlog and return the newest ground_truth, or None."""
    newest = None
    # Save and restore the timeout explicitly: setblocking(True) would clear it
    # to None, leaving the caller's socket permanently blocking.
    timeout = sock.gettimeout()
    sock.setblocking(False)
    try:
        while True:
            try:
                msg = json.loads(sock.recvfrom(65507)[0].decode("utf-8"))
            except (BlockingIOError, OSError):
                return newest
            if msg.get("type") == "ground_truth":
                newest = msg
    finally:
        sock.settimeout(timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14650, help="14650 + instance")
    parser.add_argument("--count", type=int, default=5, help="messages to print")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--latest", action="store_true",
                        help="poll on a timer, draining to the newest message")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="poll period in seconds for --latest")
    parser.add_argument("--arm-cmd", metavar="J0,J1,...",
                        help="stream this arm position command (radians) while "
                             "listening, one per ground_truth received")
    args = parser.parse_args()

    target = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.timeout)

    def send(payload: dict) -> None:
        sock.sendto(json.dumps(payload).encode("utf-8"), target)

    send({"v": SCHEMA_VERSION, "type": "subscribe"})
    arm_values = [float(v) for v in args.arm_cmd.split(",")] if args.arm_cmd else None
    arm_seq = 0

    def show(msg: dict) -> None:
        nonlocal arm_seq
        if arm_values is not None:
            arm_seq += 1
            send({"v": SCHEMA_VERSION, "type": "arm_cmd", "seq": arm_seq,
                  "mode": "position", "values": arm_values,
                  "state_time": msg["time"]})
        # Ground truth arrives in PX4 frames (NED / FRD), so it never disagrees
        # with what PX4 itself sees.
        line = (f"t={msg['time']:8.3f} pos_ned={_fmt(msg['pos_ned'])} "
                f"vel_ned={_fmt(msg['vel_ned'])} q={_fmt(msg['q_frd_ned'])} "
                f"act={_fmt(msg['actuators'])}")
        arm = msg.get("arm")
        if arm is not None:
            # Present only for a model with an arm. prop_clearance < 0 means an
            # arm capsule is inside a propeller disc: reported, never blocked.
            age = "-" if arm["cmd_age"] is None else f"{arm['cmd_age']:.3f}"
            clearance = arm["prop_clearance"]
            line += (f" arm_age={age}{' STALE' if arm['cmd_stale'] else ''}"
                     f" clearance={'-' if clearance is None else f'{clearance:+.3f}'}")
        print(line)

    try:
        for _ in range(args.count):
            if args.latest:
                time.sleep(args.interval)
                msg = drain_to_latest(sock)
                if msg is None:
                    msg = json.loads(sock.recvfrom(65507)[0].decode("utf-8"))
            else:
                msg = json.loads(sock.recvfrom(65507)[0].decode("utf-8"))
            show(msg)
    except (TimeoutError, socket.timeout):
        print(f"no ground_truth within {args.timeout:g} s -- "
              f"is the simulator running on {args.host}:{args.port}?")
        return 1
    finally:
        send({"v": SCHEMA_VERSION, "type": "unsubscribe"})
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
