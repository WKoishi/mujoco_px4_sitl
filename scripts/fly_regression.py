#!/usr/bin/env python3
"""Fly the Phase 5 regression profile: arm -> takeoff -> square -> land.

Drives PX4 over MAVLink against a running ./run_sitl.sh. Flies the profile,
reports corner errors, and stops there: **hover accuracy is not measured here.**
Run scripts/hover_error.py on the resulting ulog for that. Measuring hover from
a live MAVLink stream needs the same datum correction that script already does
properly, and two ways to compute one number is how the wrong one gets quoted.

Deliberately standalone: imports nothing from this repository, so it also works
against a PX4 driven by something else.

Usage:
    ./scripts/run_sitl.sh &                 # in another terminal
    python scripts/fly_regression.py
    python scripts/hover_error.py <the ulog it names>
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from pymavlink import mavutil

# EKF2 flags that must all be set before arming. PX4 v1.17 publishes
# ESTIMATOR_STATUS, not EKF_STATUS_REPORT -- gating on the latter waits forever
# while the vehicle sits there perfectly ready to fly.
EST_WANT = (
    mavutil.mavlink.ESTIMATOR_ATTITUDE
    | mavutil.mavlink.ESTIMATOR_POS_HORIZ_ABS
    | mavutil.mavlink.ESTIMATOR_POS_VERT_ABS
)

# Offboard is custom main mode 6. The setpoint stream must be live *before* the
# switch and must keep running, or PX4 refuses the mode and falls back.
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0
SETPOINT_HZ = 20.0


def command(m, cmd, *params):
    m.mav.command_long_send(
        m.target_system, m.target_component, cmd, 0,
        *(list(params) + [0.0] * (7 - len(params))),
    )


def ack(m, timeout=5.0):
    msg = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
    return "none" if msg is None else mavutil.mavlink.enums[
        "MAV_RESULT"][msg.result].name


def setpoint(m, north, east, down):
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000_1111_1111_1000,  # position only; everything else ignored
        north, east, down, 0, 0, 0, 0, 0, 0, 0, 0,
    )


class Link:
    """MAVLink link with the latest local position, pumped on a clock."""

    def __init__(self, m):
        self.m = m
        self.pos = None
        self.home_alt = None

    def pump(self, seconds, sender=None):
        """Service the link for ``seconds``, sending setpoints at SETPOINT_HZ."""
        end = time.time() + seconds
        last = 0.0
        while time.time() < end:
            now = time.time()
            if sender and now - last >= 1.0 / SETPOINT_HZ:
                sender()
                last = now
            msg = self.m.recv_match(
                type=["LOCAL_POSITION_NED", "HOME_POSITION", "STATUSTEXT"],
                blocking=True, timeout=0.02,
            )
            if msg is None:
                continue
            kind = msg.get_type()
            if kind == "STATUSTEXT":
                print(f"  PX4: {msg.text}", flush=True)
            elif kind == "HOME_POSITION":
                self.home_alt = msg.altitude * 1e-3
            else:
                self.pos = (msg.x, msg.y, msg.z)

    def wait_ekf(self, timeout):
        start = time.time()
        while time.time() - start < timeout:
            msg = self.m.recv_match(type="ESTIMATOR_STATUS", blocking=True, timeout=5)
            if msg and (msg.flags & EST_WANT) == EST_WANT:
                print(f"EKF2 converged, flags=0x{msg.flags:04x} at "
                      f"t+{time.time() - start:.1f}s", flush=True)
                return True
        return False


def fly(link, args):
    m = link.m
    # A previous flight leaves PX4 in Land, and arming from Land is refused with
    # "Resolve system health failures first" -- naming neither the mode nor a
    # sensor. Hold is AUTO (4) / LOITER (3), the mode a fresh boot sits in.
    command(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
    print(f"Hold {ack(m)}", flush=True)
    link.pump(1.0)

    print("arming", flush=True)
    command(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)
    result = ack(m)
    print(f"  {result}", flush=True)
    if result != "MAV_RESULT_ACCEPTED":
        # Carrying on would send a takeoff that is also ACCEPTED and then report
        # "did not climb", which reads as a control failure.
        print("arming refused; the PX4 console says why", flush=True)
        return None
    link.pump(2.0)

    # MAV_CMD_NAV_TAKEOFF's param7 is **AMSL**, not relative to home. Passing a
    # relative altitude puts the target below home (SITL home is 488 m), and
    # navigator rejects it with "Already higher than takeoff altitude" -- after
    # returning MAV_RESULT_ACCEPTED. The vehicle then sits armed on the ground
    # until "Disarmed by auto preflight disarming", which reads like a control
    # failure rather than a bad argument. MIS_TAKEOFF_ALT is the relative one.
    if link.home_alt is None:
        print("no HOME_POSITION received; cannot resolve AMSL takeoff altitude",
              flush=True)
        return None
    target_amsl = link.home_alt + args.alt
    print(f"takeoff to {args.alt} m above home "
          f"(home {link.home_alt:.1f} m AMSL, param7 {target_amsl:.1f})", flush=True)
    command(m, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"),
            float("nan"), float("nan"), target_amsl)
    print(f"  {ack(m)}", flush=True)
    link.pump(args.climb)
    print(f"  local position {_fmt(link.pos)}", flush=True)
    if link.pos is None or link.pos[2] > -0.5 * args.alt:
        print("did not climb; aborting before the square", flush=True)
        return None

    print("streaming setpoints, then switching to Offboard", flush=True)
    for _ in range(int(2.0 * SETPOINT_HZ)):
        setpoint(m, 0.0, 0.0, -args.alt)
        time.sleep(1.0 / SETPOINT_HZ)
    command(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1.0,
            PX4_CUSTOM_MAIN_MODE_OFFBOARD, 0.0)
    print(f"  Offboard {ack(m)}", flush=True)

    leg = args.leg
    corners = [(leg, 0.0), (leg, leg), (0.0, leg), (0.0, 0.0)]
    errors = []
    for i, (north, east) in enumerate(corners, start=1):
        print(f"leg {i}: -> N={north} E={east}", flush=True)
        link.pump(args.corner_wait,
                  sender=lambda n=north, e=east: setpoint(m, n, e, -args.alt))
        err = math.hypot(link.pos[0] - north, link.pos[1] - east)
        errors.append(err)
        print(f"  {_fmt(link.pos)}  corner error {err:.3f} m", flush=True)

    print(f"holding hover for {args.settle:g}s (logged, measured from the ulog)",
          flush=True)
    link.pump(args.settle, sender=lambda: setpoint(m, 0.0, 0.0, -args.alt))

    print("landing", flush=True)
    command(m, mavutil.mavlink.MAV_CMD_NAV_LAND)
    print(f"  {ack(m)}", flush=True)
    link.pump(args.land_wait)
    print(f"  final {_fmt(link.pos)}", flush=True)
    return errors


def _fmt(pos):
    return "pos=None" if pos is None else \
        f"pos_ned=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-i", "--instance", type=int, default=0,
                   help="PX4 instance; GCS port is 14550+N (default: 0)")
    p.add_argument("--alt", type=float, default=5.0, help="takeoff altitude, m")
    p.add_argument("--leg", type=float, default=5.0, help="square side, m")
    p.add_argument("--corner-wait", type=float, default=20.0,
                   help="seconds held at each corner (default: 20, matching the "
                        "baseline -- still mid-settle, not converged)")
    p.add_argument("--settle", type=float, default=60.0,
                   help="seconds of hover to log for the ulog measurement")
    p.add_argument("--climb", type=float, default=30.0, help="seconds to allow for climb")
    p.add_argument("--land-wait", type=float, default=40.0, help="seconds to allow for landing")
    p.add_argument("--ekf-timeout", type=float, default=90.0)
    args = p.parse_args()

    m = mavutil.mavlink_connection(
        f"udpin:127.0.0.1:{14550 + args.instance}", source_system=255
    )
    m.wait_heartbeat()
    print(f"heartbeat: sys {m.target_system}", flush=True)

    link = Link(m)
    if not link.wait_ekf(args.ekf_timeout):
        print("EKF2 did not converge", flush=True)
        return 1
    link.pump(1.0)  # pick up HOME_POSITION

    errors = fly(link, args)
    if errors is None:
        return 1

    print(f"\ncorner errors (m): {', '.join('%.2f' % e for e in errors)}", flush=True)
    print("Hover accuracy is NOT in the output above. Measure it from the ulog:",
          flush=True)
    print("  python scripts/hover_error.py "
          "../PX4-Autopilot/build/px4_sitl_default/rootfs/log/*/*.ulg", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
