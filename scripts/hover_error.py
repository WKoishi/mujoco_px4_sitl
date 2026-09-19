#!/usr/bin/env python3
"""Hover accuracy from a ulog: EKF2 estimate against MuJoCo ground truth.

Compares `vehicle_local_position` with `vehicle_local_position_groundtruth` over
the settled part of each hover, which is how IMPLEMENTATION_PLAN.md phase 5's
baseline was measured.

Two corrections make the difference between a real number and a plausible one:

**Each topic latches its own datum.** `ref_lat` / `ref_lon` / `ref_alt` differ
between the two -- EKF2 picks its own once it has a global fix, ground truth uses
the simulator's home. Comparing raw x/y/z folds that offset into the error, and
it reads as a systematic bias of several decimetres. The tell is the median
sitting almost exactly on the max: GPS noise gives a median well below the max, a
constant offset does not.

**Settling is not error.** Sampled from the moment the vehicle stops moving, the
climb's overshoot dominates the max. The baseline let hover settle first, so this
drops the first --settle-skip seconds of each still stretch.

Usage:
    python scripts/hover_error.py path/to/log.ulg
    python scripts/hover_error.py ../PX4-Autopilot/build/px4_sitl_default/rootfs/log/*/*.ulg
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pyulog import ULog

M_PER_DEG_LAT = 111320.0
EST = "vehicle_local_position"
TRUTH = "vehicle_local_position_groundtruth"


def series(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d
    raise SystemExit(f"topic {name!r} not in log -- logger profile too narrow?")


def datum(d):
    """First finite non-zero ref_lat / ref_lon / ref_alt of a topic."""
    out = []
    for field in ("ref_lat", "ref_lon", "ref_alt"):
        v = np.asarray(d.data[field], dtype=np.float64)
        v = v[np.isfinite(v) & (v != 0.0)]
        if v.size == 0:
            raise SystemExit(f"{d.name}.{field} never set; no global reference")
        out.append(float(v[0]))
    return out


def settled(t, mask, skip):
    """Indices in ``mask``, minus the first ``skip`` seconds of each stretch."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    keep = []
    run_start = t[idx[0]]
    for i, j in enumerate(idx):
        if i > 0 and t[j] - t[idx[i - 1]] > 1.0:  # gap: a new stretch
            run_start = t[j]
        if t[j] - run_start >= skip:
            keep.append(j)
    return np.asarray(keep, dtype=int)


def analyse(path, args):
    ulog = ULog(path, [EST, TRUTH])
    est, truth = series(ulog, EST), series(ulog, TRUTH)
    t = np.asarray(est.data["timestamp"], dtype=np.float64) * 1e-6
    t_truth = np.asarray(truth.data["timestamp"], dtype=np.float64) * 1e-6

    print(f"{path}")
    print(f"  {len(t)} estimate samples, {len(t_truth)} truth samples, "
          f"spanning {t[0]:.1f}..{t[-1]:.1f} s")

    e_lat, e_lon, e_alt = datum(est)
    t_lat, t_lon, t_alt = datum(truth)
    off_n = (t_lat - e_lat) * M_PER_DEG_LAT
    off_e = (t_lon - e_lon) * M_PER_DEG_LAT * np.cos(np.radians(e_lat))
    off_u = t_alt - e_alt
    print(f"  estimate datum  {e_lat:.8f} {e_lon:.8f} {e_alt:.3f} m")
    print(f"  truth datum     {t_lat:.8f} {t_lon:.8f} {t_alt:.3f} m")
    print(f"  datum offset    N {off_n:+.3f}  E {off_e:+.3f}  U {off_u:+.3f} m "
          f"-- removed")

    # Truth resampled onto the estimate's timestamps, then shifted into the
    # estimate's frame.
    tx = np.interp(t, t_truth, truth.data["x"]) + off_n
    ty = np.interp(t, t_truth, truth.data["y"]) + off_e
    tz = np.interp(t, t_truth, truth.data["z"]) - off_u

    ez = np.asarray(est.data["z"], dtype=np.float64)
    dh = np.hypot(np.asarray(est.data["x"]) - tx, np.asarray(est.data["y"]) - ty)
    dv = np.abs(ez - tz)

    airborne = ez < -args.min_alt
    speed = np.hypot(est.data["vx"], est.data["vy"])
    hover = airborne & (speed < args.max_speed)
    keep = settled(t, hover, args.settle_skip)
    print(f"  {int(airborne.sum())} airborne, {int(hover.sum())} of those still, "
          f"{len(keep)} after dropping {args.settle_skip:g}s per stretch")
    if len(keep) < 50:
        print("  too few settled samples to report -- lower --settle-skip or fly longer")
        return 1

    span = t[keep[-1]] - t[keep[0]]
    print(f"\n  steady-state hover, {span:.1f} s of samples:")
    print(f"    horizontal   max {dh[keep].max():.3f} m   "
          f"median {np.median(dh[keep]):.3f} m")
    print(f"    vertical     max {dv[keep].max():.3f} m   "
          f"median {np.median(dv[keep]):.3f} m")
    print("  whole airborne segment:")
    print(f"    horizontal   max {dh[airborne].max():.3f} m")
    print(f"    vertical     max {dv[airborne].max():.3f} m")

    # Judge against what PX4's own synthetic GPS can deliver, not against zero:
    # sensor_gps_sim adds sigma = 0.2 m horizontal, 0.5 m vertical, and
    # EKF2_HGT_REF defaults to GNSS (plan phase 5).
    print("\n  reference: sensor_gps_sim noise is sigma 0.2 m horizontal, "
          "0.5 m vertical")
    if np.median(dv[keep]) > 0.1 or np.median(dh[keep]) > 0.1:
        print("  NOTE median error above 0.1 m. If median is close to max, "
              "suspect a datum or frame error rather than tuning.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ulog", nargs="+", help="ulog file(s)")
    p.add_argument("--settle-skip", type=float, default=10.0,
                   help="seconds discarded at the start of each still stretch "
                        "(default: 10; the climb transient reads as bias otherwise)")
    p.add_argument("--min-alt", type=float, default=1.0,
                   help="metres above the local origin to count as airborne")
    p.add_argument("--max-speed", type=float, default=0.15,
                   help="horizontal speed below which the vehicle counts as hovering")
    args = p.parse_args()

    status = 0
    for path in args.ulog:
        status |= analyse(path, args)
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
