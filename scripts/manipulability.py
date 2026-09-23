#!/usr/bin/env python3
"""End-effector manipulability over the reachable workspace, as a point cloud.

Answers "can this arm hold an arbitrary pose at an arbitrary point". That is a
property of the Jacobian, not of the dynamics, so nothing here involves PX4, a
flight, or even a physics step -- only forward kinematics.

The multirotor contributes **yaw only**. Its three translations are ignored
because a free-flying base puts the workspace wherever it likes, and roll and
pitch are ignored because neither is a steady state in hover. So the movable set
is ``yaw`` plus the arm joints, and ``--freeze`` locks any subset of them.

Two properties of yaw make the base-frame cloud complete rather than a slice,
and both are asserted in ``tests/test_manipulability.py``:

* ``w`` is invariant to yaw -- it is a rigid rotation of the whole chain.
* the reachable set *in the base frame* does not depend on yaw either; yaw only
  rotates the cloud about the base z axis.

So freezing yaw changes ``w`` while leaving the cloud's shape untouched, and
sampling yaw buys nothing. ``--revolve`` sweeps copies around z if the swept
volume is what you want to look at.

**``w`` compares only within one task dimension.** It is an ``m x m``
determinant, so dropping a joint removes a factor and generally makes the number
*larger*. Across configurations compare rank deficiency, which is dimensionless;
reserve ``w`` for ranking poses under a fixed task.

The headline number is the fraction of reachable voxels that no sampled
configuration reaches at full task rank. Those are the voxels where the task
cannot be executed at all, however the arm approaches them.

**Use ``--slice`` whenever you look at a cloud.** A solid one renders only its own
shell: the outer voxels hide every interior voxel, so the whole-workspace view
says nothing about the middle. A slab one voxel thick is a cross-section.

Standalone on purpose: imports nothing from this repository, so it runs against
any MJCF with an end-effector site (``MODELING_CONVENTIONS.md`` section 8).

Usage::

    python scripts/manipulability.py <model>.xml --task pose --slice y --view
    python scripts/manipulability.py <model>.xml --task pose --freeze arm_joint3
    python scripts/manipulability.py <model>.xml --task tool-axis --save-npz out.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger("manipulability")

EE_SITE = "ee"
BASE_BODY = "base_link"
YAW_LABEL = "base_yaw"

# Task row sets, as indices into the stacked [jacp; jacr] 6 x nv Jacobian.
#
# "tool-axis" drops rotation about the tool's own z: for a gripper that is
# usually free. It is built per-configuration because the two rows depend on
# where the tool points, so it lives in _task_rows rather than here.
TASKS = {
    "position": (0, 1, 2),
    "pose": (0, 1, 2, 3, 4, 5),
    "tool-axis": None,  # built per configuration
}

# Rank tolerance, relative to the largest singular value. Measured on the X8
# arm: at full rank the worst sigma_min/sigma_max over 2000 random poses is
# 1.5e-05, while a genuinely deficient Jacobian lands at ~1e-16. Four orders of
# margin either side, so no defensible value inside the gap changes the verdict.
RANK_RTOL = 1e-9

# Categorical slots 1 and 2, validated all-pairs in both light and dark
# (dataviz skill). Blue reads as "full rank", orange as "deficient"; the pair
# stays separable under every simulated CVD.
RGBA_FULL = (0.224, 0.529, 0.898, 1.0)   # #3987e5
RGBA_DEFICIENT = (0.851, 0.349, 0.149, 1.0)  # #d95926

# mjv_initGeom writes into a fixed-size scene buffer. The passive viewer's own
# scene is sized _Simulate.MAX_GEOM; leave room for the model's real geoms.
VIEWER_MAX_GEOM = 100_000


# --- the movable set ------------------------------------------------------


@dataclass
class Movable:
    """The joints that move, and the Jacobian columns they own.

    ``labels`` is parallel to ``dofs`` so a frozen joint can be named in the
    report. Position DoFs are never here: a free-flying base can put the
    workspace anywhere, so including them would make every point full-rank and
    measure nothing.
    """

    dofs: list[int]
    labels: list[str]
    frozen: list[str]
    yaw_dof: int | None
    arm_joints: list[int]   # joint ids, movable only
    all_arm_joints: list[int]  # joint ids, including frozen ones

    @property
    def n(self) -> int:
        return len(self.dofs)


def build_movable(model: mujoco.MjModel, freeze: list[str]) -> Movable:
    """Collect yaw plus every scalar arm joint, minus whatever ``freeze`` names.

    Raises on an unknown name in ``freeze``: a typo would otherwise silently
    leave the joint movable and the two runs being compared would differ by
    nothing at all.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    if base_id < 0:
        raise ValueError(f"model has no body named {BASE_BODY!r}")

    dofs: list[int] = []
    labels: list[str] = []
    frozen: list[str] = []
    yaw_dof: int | None = None
    arm: list[int] = []
    all_arm: list[int] = []
    known: list[str] = [YAW_LABEL]

    # Yaw is the free joint's 6th DoF: MuJoCo orders a freejoint's velocity as
    # 3 translations then 3 rotations about world x, y, z. Verified against a
    # finite difference in tests/test_manipulability.py -- the ordering is the
    # kind of thing that is wrong silently.
    # int() on every jnt_type: it is an np.int32, and `np.int32 in (enum, enum)`
    # is False even where `np.int32 == enum` is True. That silently emptied the
    # movable set once already -- the run reported a 1-DoF vehicle and finished
    # without an error.
    free_t = int(mujoco.mjtJoint.mjJNT_FREE)
    scalar_t = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))

    for j in range(model.njnt):
        if int(model.jnt_type[j]) != free_t:
            continue
        if model.jnt_bodyid[j] == base_id:
            yaw_dof = int(model.jnt_dofadr[j]) + 5
        break

    if yaw_dof is not None and YAW_LABEL not in freeze:
        dofs.append(yaw_dof)
        labels.append(YAW_LABEL)
    elif yaw_dof is not None:
        frozen.append(YAW_LABEL)

    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in scalar_t:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        all_arm.append(j)
        known.append(name)
        if name in freeze:
            frozen.append(name)
            continue
        dofs.append(int(model.jnt_dofadr[j]))
        labels.append(name)
        arm.append(j)

    unknown = sorted(set(freeze) - set(known))
    if unknown:
        raise ValueError(
            f"--freeze names no such joint: {', '.join(unknown)}. "
            f"Available: {', '.join(known)}"
        )
    if not dofs:
        raise ValueError("every movable joint is frozen; nothing to sample")
    return Movable(dofs, labels, frozen, yaw_dof, arm, all_arm)


def joint_limits(model: mujoco.MjModel, joints: list[int]) -> NDArray[np.float64]:
    """Sampling bounds per joint, refusing the 0/0 export.

    MuJoCo compiles ``lower == upper == 0`` as *unlimited*
    (``MODELING_CONVENTIONS.md`` section 6.2), so a raw SolidWorks export would
    otherwise be sampled over a range of exactly zero -- one configuration,
    reported as a workspace.
    """
    lim = np.zeros((len(joints), 2))
    bad: list[str] = []
    for i, j in enumerate(joints):
        lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
        if not model.jnt_limited[j] or hi <= lo:
            bad.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"#{j}")
        lim[i] = (lo, hi)
    if bad:
        raise ValueError(
            "these joints have no usable range (MuJoCo reads 0/0 as unlimited, "
            f"so the export's default is a free joint, not a locked one): {', '.join(bad)}"
        )
    return lim


# --- the metric -----------------------------------------------------------


def _task_rows(
    jacp: NDArray[np.float64], jacr: NDArray[np.float64], task: str,
    site_xmat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Stack the task's rows out of the 3 x nv position and rotation Jacobians."""
    if task == "tool-axis":
        # Keep position, plus the two angular directions orthogonal to the tool
        # axis. Rotation *about* the axis is what a gripper does not care about,
        # and on this arm it is exactly what the wrist joint provides -- so
        # including it would credit the wrist for a freedom the task discards.
        tool = site_xmat.reshape(3, 3)[:, 2]
        e1 = np.cross(tool, (1.0, 0.0, 0.0))
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(tool, (0.0, 1.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(tool, e1)
        return np.vstack([jacp, e1 @ jacr, e2 @ jacr])
    return np.vstack([jacp, jacr])[list(TASKS[task]), :]


def manipulability(jac: NDArray[np.float64]) -> tuple[float, bool]:
    """Yoshikawa's measure and whether the task rank is met.

    ``w = sqrt(det(J J^T))`` -- an ``m x m`` determinant, so it is zero exactly
    when the Jacobian cannot span the task. Computing it as a *product of
    singular values* instead is a trap: ``svd`` of a 6 x 5 matrix returns five
    values whose product is happily nonzero, which reports a rank-5 arm as
    capable of a 6D task.
    """
    gram = jac @ jac.T
    det = float(np.linalg.det(gram))
    w = float(np.sqrt(max(det, 0.0)))
    sv = np.linalg.svd(jac, compute_uv=False)
    full = bool(sv.size >= jac.shape[0] and sv[-1] > RANK_RTOL * sv[0])
    return w, full


# --- sampling -------------------------------------------------------------


@dataclass
class Cloud:
    """Voxelised workspace. One entry per occupied voxel, in the base frame.

    ``w`` is the **best** value any sampled configuration achieved in that
    voxel, not the mean: the question is whether the arm *can* work there, and
    one good approach is enough. Averaging would smear a reachable voxel toward
    zero just because most ways of reaching it are awkward.
    """

    pos: NDArray[np.float64]      # (n, 3) voxel centres, base frame
    w: NDArray[np.float64]        # (n,) best w in that voxel
    full_rank: NDArray[np.bool_]  # (n,) any configuration met task rank
    samples: int
    task: str
    voxel: float
    movable: Movable

    @property
    def deficient_fraction(self) -> float:
        if not len(self.full_rank):
            return 0.0
        return float(np.mean(~self.full_rank))


def sample(
    model: mujoco.MjModel, task: str, movable: Movable, samples: int,
    voxel: float, seed: int, progress: bool = True,
) -> Cloud:
    """Sample joint space uniformly, reduce to voxels, keep the best w per voxel.

    Yaw is held at zero even when movable. It is a rigid rotation of the whole
    chain, so it changes neither ``w`` nor the base-frame reachable set -- only
    which world direction the cloud faces. Sampling it would spend the budget
    re-measuring one cloud from many angles.
    """
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if site < 0:
        raise ValueError(
            f"model has no site named {EE_SITE!r}; the end-effector reference "
            "is optional in the conversion (MODELING_CONVENTIONS.md section 8) "
            "but required here"
        )
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    limits = joint_limits(model, movable.arm_joints)
    qadr = np.array([model.jnt_qposadr[j] for j in movable.arm_joints], dtype=int)

    rng = np.random.default_rng(seed)
    cols = movable.dofs
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pts = np.empty((samples, 3))
    ws = np.empty(samples)
    ok = np.empty(samples, dtype=bool)

    # Frozen joints sit at qpos 0, which is the model's own reference pose. A
    # frozen joint parked somewhere else is a different vehicle, so 0 is the
    # only defensible choice without a second flag.
    mujoco.mj_resetData(model, data)
    home = data.qpos.copy()
    mujoco.mj_kinematics(model, data)
    # The base never moves during sampling (only arm qpos is written), so its
    # pose is the home pose and one read is enough.
    origin = data.xpos[base_id].copy()
    t0 = time.monotonic()
    step = max(samples // 10, 1)

    for i in range(samples):
        data.qpos[:] = home
        data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
        # Kinematics plus comPos: mj_jacSite reads xpos/xmat and the DoF frames,
        # and comPos is what fills the latter. Without it the Jacobian is stale
        # by one configuration -- a silent error, since the shape stays right.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jac = _task_rows(jacp, jacr, task, data.site_xmat[site])[:, cols]
        ws[i], ok[i] = manipulability(jac)
        pts[i] = data.site_xpos[site]
        if progress and (i + 1) % step == 0:
            _log.info("  sampled %d/%d", i + 1, samples)

    # Into the base frame, so a cloud is comparable across base poses and the
    # voxel grid does not depend on where the vehicle happens to sit.
    local = pts - origin

    keys = np.floor(local / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    best = np.zeros(len(uniq))
    np.maximum.at(best, inv, ws)
    anyfull = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(anyfull, inv, ok)
    centres = (uniq + 0.5) * voxel

    _log.info(
        "%d samples -> %d voxels of %.0f mm in %.1f s",
        samples, len(uniq), voxel * 1e3, time.monotonic() - t0,
    )
    return Cloud(centres, best, anyfull, samples, task, voxel, movable)


# --- reporting ------------------------------------------------------------


def report(cloud: Cloud) -> str:
    """The numbers to quote, and the caveat that keeps them from being misread."""
    m = cloud.movable
    lines = [
        f"task            {cloud.task}",
        f"movable         {', '.join(m.labels)}  ({m.n} DoF)",
        f"frozen          {', '.join(m.frozen) if m.frozen else '(none)'}",
        f"samples         {cloud.samples}",
        f"voxel           {cloud.voxel * 1e3:.0f} mm",
        f"reachable       {len(cloud.pos)} voxels",
    ]
    frac = cloud.deficient_fraction
    lines.append(
        f"rank-deficient  {int(np.sum(~cloud.full_rank))} voxels "
        f"({frac * 100:.1f}% of reachable)"
    )
    live = cloud.w[cloud.full_rank]
    if live.size:
        lines += [
            f"w (full rank)   median {np.median(live):.4e}  max {live.max():.4e}",
        ]
    else:
        lines.append("w (full rank)   -- no voxel reaches full task rank")

    if frac > 0.99:
        lines.append(
            "\nEvery reachable voxel is rank-deficient. With "
            f"{m.n} movable DoF against this task, that follows from counting "
            "alone -- no pose anywhere can span it."
        )
    lines.append(
        "\nw compares only within one task and one movable set: it is an m x m "
        "determinant, so removing a joint drops a factor and usually makes the "
        "number LARGER. Compare rank deficiency across configurations."
    )
    return "\n".join(lines)


# --- rendering ------------------------------------------------------------


def _shade(w: float, wmax: float) -> tuple[float, float, float, float]:
    """One-hue blue ramp over the full-rank voxels, lightening with w.

    Steps 550 -> 100 of the reference blue ramp. Two departures from the usual
    "light means near zero", both forced by the surface: MuJoCo's scene is dark,
    so the light end is what stands out, and the ramp stops at step 550 rather
    than running to 700 because anything darker disappears into the background
    (700 measures 1.4:1 against it). Validated as an ordinal ramp against
    #1a1a19, which is why intermediate step 450 is skipped -- it sits 0.047 in
    lightness from 400 and the two read as one colour.
    """
    dark = np.array([0.110, 0.361, 0.671])   # #1c5cab, step 550
    light = np.array([0.804, 0.886, 0.984])  # #cde2fb, step 100
    t = 0.0 if wmax <= 0.0 else float(np.clip(w / wmax, 0.0, 1.0))
    # sqrt so the low end is not crushed: w spans decades, and a linear map
    # would paint all but the best poses the same dark blue.
    c = dark + (light - dark) * np.sqrt(t)
    return float(c[0]), float(c[1]), float(c[2]), 1.0


def draw(
    scene: mujoco.MjvScene, cloud: Cloud, origin: NDArray[np.float64],
    radius: float, revolve: int = 1, deficient_only: bool = False,
    slice_axis: str | None = None, slice_at: float = 0.0,
    slice_width: float | None = None,
) -> int:
    """Add one sphere per voxel to ``scene``, returning how many were drawn.

    Colour carries the verdict: orange where no configuration met the task rank,
    blue shaded by w where one did. Those are two different questions, so they
    get two different hues rather than two ends of one ramp.

    A solid cloud shows only its own shell -- the outer voxels hide every
    interior one, so a plot of the whole workspace says nothing about the middle.
    ``slice_axis`` keeps a slab one voxel thick, which is the honest way to look
    inside: it is a cross-section, not a projection.
    """
    live = cloud.w[cloud.full_rank]
    wmax = float(live.max()) if live.size else 0.0
    size = np.array([radius, radius, radius])
    eye = np.eye(3).flatten()
    drawn = 0
    angles = np.linspace(0.0, 2.0 * np.pi, revolve, endpoint=False)

    keep = np.ones(len(cloud.pos), dtype=bool)
    if slice_axis is not None:
        col = {"x": 0, "y": 1, "z": 2}[slice_axis]
        half = (slice_width if slice_width is not None else cloud.voxel) / 2.0
        keep &= np.abs(cloud.pos[:, col] - slice_at) <= half

    for ang in angles:
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for i in range(len(cloud.pos)):
            if not keep[i]:
                continue
            full = bool(cloud.full_rank[i])
            if deficient_only and full:
                continue
            if scene.ngeom >= scene.maxgeom:
                _log.warning(
                    "scene full at %d geoms; raise --voxel or lower --revolve",
                    scene.ngeom,
                )
                return drawn
            rgba = _shade(cloud.w[i], wmax) if full else RGBA_DEFICIENT
            pos = rot @ cloud.pos[i] + origin
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                size, pos, eye, np.array(rgba, dtype=np.float32),
            )
            scene.ngeom += 1
            drawn += 1
    return drawn


def view(model: mujoco.MjModel, cloud: Cloud, args: argparse.Namespace) -> None:
    """Passive viewer with the cloud in ``user_scn``, so the mouse still orbits.

    Passive is the right mode for the same reason ``viewer.py`` gives: it hands
    us the sync point instead of running its own loop. Here there is no physics
    to drive at all -- the arm is parked and the cloud is static -- so the loop
    only keeps the window alive.
    """
    import mujoco.viewer  # imported lazily: needs GL

    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    mujoco.mj_resetData(model, data)
    if args.pose:
        for name, val in args.pose:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"--pose names no such joint: {name}")
            data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)
    origin = data.xpos[base_id].copy()

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as handle:
        scene = handle.user_scn
        if scene is None:  # pragma: no cover - only on a build without user_scn
            raise RuntimeError("this mujoco build exposes no user_scn")
        drawn = draw(
            scene, cloud, origin, args.radius,
            revolve=args.revolve, deficient_only=args.deficient_only,
            slice_axis=args.slice, slice_at=args.slice_at,
            slice_width=args.slice_width,
        )
        _log.info(
            "drew %d spheres (%d voxels x %d revolutions); "
            "drag to orbit, scroll to zoom",
            drawn, len(cloud.pos), args.revolve,
        )
        handle.sync()
        while handle.is_running():
            time.sleep(0.05)


# --- CLI ------------------------------------------------------------------


def _pose_pair(text: str) -> tuple[str, float]:
    name, _, val = text.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"--pose wants joint=radians, got {text!r}")
    return name, float(val)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Freezing a joint and comparing w against the unfrozen run is the "
            "one comparison this tool cannot support; compare rank deficiency."
        ),
    )
    p.add_argument("model", type=Path, help="MJCF with an 'ee' site")
    p.add_argument(
        "--task", default="pose", choices=sorted(TASKS),
        help="position: 3D point only. pose: full 6D. tool-axis: 5D, position "
             "plus tool direction, free about the tool's own axis (default: pose)",
    )
    p.add_argument(
        "--freeze", default=[], nargs="+", metavar="JOINT",
        help=f"joints to lock at qpos 0; '{YAW_LABEL}' locks the vehicle heading",
    )
    p.add_argument("--samples", type=int, default=40_000)
    p.add_argument("--voxel", type=float, default=0.04, help="metres (default 0.04)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    p.add_argument("--radius", type=float, default=0.008, help="sphere radius, m")
    p.add_argument(
        "--revolve", type=int, default=1, metavar="N",
        help="draw N copies around the yaw axis to show the swept volume; w does "
             "not depend on yaw, so this is presentation only (default 1)",
    )
    p.add_argument(
        "--deficient-only", action="store_true",
        help="draw only the voxels that never reach full task rank",
    )
    p.add_argument(
        "--slice", choices=("x", "y", "z"),
        help="keep only a slab normal to this base-frame axis. A solid cloud "
             "shows only its shell, so this is the way to see the interior",
    )
    p.add_argument(
        "--slice-at", type=float, default=0.0, metavar="M",
        help="slab centre along --slice, metres from the base origin (default 0)",
    )
    p.add_argument(
        "--slice-width", type=float, metavar="M",
        help="slab thickness (default: one voxel, i.e. a true cross-section)",
    )
    p.add_argument(
        "--pose", type=_pose_pair, nargs="+", default=[], metavar="JOINT=RAD",
        help="park the arm at this pose for the viewer's model render",
    )
    p.add_argument("--save-npz", type=Path, help="write the cloud for later plotting")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    if not args.model.exists():
        _log.error("no such model: %s", args.model)
        return 2

    model = mujoco.MjModel.from_xml_path(str(args.model))
    try:
        movable = build_movable(model, list(args.freeze))
        cloud = sample(
            model, args.task, movable, args.samples, args.voxel, args.seed,
            progress=not args.quiet,
        )
    except ValueError as exc:
        _log.error("%s", exc)
        return 2

    print(report(cloud))

    if args.save_npz:
        np.savez(
            args.save_npz, pos=cloud.pos, w=cloud.w, full_rank=cloud.full_rank,
            task=cloud.task, voxel=cloud.voxel, samples=cloud.samples,
            movable=np.array(movable.labels), frozen=np.array(movable.frozen),
        )
        _log.info("wrote %s", args.save_npz)

    if args.view:
        view(model, cloud, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
