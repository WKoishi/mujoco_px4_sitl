"""Tests for scripts/manipulability.py.

No private geometry: every test builds a small MJCF inline, so this runs in a
fresh checkout. The focus is the failures that are **silent** -- a Jacobian
column that is quietly the wrong DoF, a rank-deficient arm reported as capable,
a 0/0 joint range sampled as a single point, and a movable set that comes back
empty because an enum comparison did not fire.

The last one is not hypothetical: ``np.int32 in (mjJNT_HINGE, mjJNT_SLIDE)`` is
False where ``==`` is True, and the first run of this tool reported the X8 as a
1-DoF vehicle and exited 0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "manipulability.py"
_spec = importlib.util.spec_from_file_location("manipulability", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mp = importlib.util.module_from_spec(_spec)
sys.modules["manipulability"] = mp
_spec.loader.exec_module(mp)


# --- fixtures -------------------------------------------------------------

# A planar arm: every hinge is about z, so the end-effector can never leave its
# plane. Position rank is therefore at most 2 and a 3D position task is
# deficient everywhere -- a rank verdict with an analytic answer.
PLANAR = """
<mujoco model="planar">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base_link" pos="0 0 0.5">
      <joint name="base_free" type="free"/>
      <geom type="box" size="0.1 0.1 0.02" mass="1"/>
      <body name="l0" pos="0.1 0 0">
        <joint name="j0" type="hinge" axis="0 0 1" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.2"/>
        <body name="l1" pos="0.3 0 0">
          <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.2"/>
          <site name="ee" pos="0.3 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Same chain, but the last joint spins about the axis the site sits on, so it
# moves the tool's orientation and nothing else. This is the X8 wrist's
# geometry, reduced to the part that matters.
SPIN_WRIST = """
<mujoco model="spin">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base_link" pos="0 0 0.5">
      <joint name="base_free" type="free"/>
      <geom type="box" size="0.1 0.1 0.02" mass="1"/>
      <body name="l0" pos="0.1 0 0">
        <joint name="j0" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.2"/>
        <body name="l1" pos="0.3 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
          <body name="wrist" pos="0.2 0 0">
            <joint name="spin" type="hinge" axis="1 0 0" range="-3 3"/>
            <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015" mass="0.05"/>
            <site name="ee" pos="0.05 0 0" zaxis="1 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

UNLIMITED = """
<mujoco model="unlimited">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base_link" pos="0 0 0.5">
      <joint name="base_free" type="free"/>
      <geom type="box" size="0.1 0.1 0.02" mass="1"/>
      <body name="l0" pos="0.1 0 0">
        <joint name="j0" type="hinge" axis="0 0 1" range="0 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.2"/>
        <site name="ee" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def planar() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(PLANAR)


@pytest.fixture
def spin() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(SPIN_WRIST)


# --- the movable set ------------------------------------------------------


def test_every_hinge_is_found(planar: mujoco.MjModel) -> None:
    """The enum-comparison trap: an empty movable set must not be possible.

    ``np.int32 in (enum, enum)`` is False where ``==`` is True, which once left
    only base_yaw movable and produced a clean-looking report.
    """
    mv = mp.build_movable(planar, [])
    assert mv.labels == [mp.YAW_LABEL, "j0", "j1"]
    assert mv.n == 3
    assert mv.arm_joints, "no arm joint was recognised as movable"


def test_yaw_dof_is_the_sixth_free_dof(planar: mujoco.MjModel) -> None:
    """Verify by finite difference, not by trusting the DoF ordering.

    A freejoint's velocity is 3 translations then 3 rotations about world x, y,
    z. Picking the wrong index yields a plausible cloud with the wrong column.
    """
    mv = mp.build_movable(planar, [])
    data = mujoco.MjData(planar)
    site = mujoco.mj_name2id(planar, mujoco.mjtObj.mjOBJ_SITE, "ee")
    data.qpos[planar.jnt_qposadr[1]] = 0.6  # bend, so the ee is off the yaw axis
    mujoco.mj_kinematics(planar, data)
    mujoco.mj_comPos(planar, data)
    before = data.site_xpos[site].copy()
    jacp = np.zeros((3, planar.nv))
    jacr = np.zeros((3, planar.nv))
    mujoco.mj_jacSite(planar, data, jacp, jacr, site)

    eps = 1e-6
    rot = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rot, np.array([0.0, 0.0, 1.0]), eps)
    turned = np.zeros(4)
    mujoco.mju_mulQuat(turned, rot, data.qpos[3:7])
    data.qpos[3:7] = turned
    mujoco.mj_kinematics(planar, data)
    fd = (data.site_xpos[site] - before) / eps

    assert mv.yaw_dof is not None
    np.testing.assert_allclose(jacp[:, mv.yaw_dof], fd, atol=1e-4)
    np.testing.assert_allclose(jacr[:, mv.yaw_dof], [0.0, 0.0, 1.0], atol=1e-9)


def test_freeze_removes_the_column_and_the_sampling(planar: mujoco.MjModel) -> None:
    mv = mp.build_movable(planar, ["j1"])
    assert mv.labels == [mp.YAW_LABEL, "j0"]
    assert mv.frozen == ["j1"]
    assert [planar.jnt_qposadr[j] for j in mv.arm_joints] == [
        planar.jnt_qposadr[mujoco.mj_name2id(planar, mujoco.mjtObj.mjOBJ_JOINT, "j0")]
    ]


def test_freeze_yaw(planar: mujoco.MjModel) -> None:
    mv = mp.build_movable(planar, [mp.YAW_LABEL])
    assert mp.YAW_LABEL not in mv.labels
    assert mv.frozen == [mp.YAW_LABEL]


def test_unknown_freeze_name_raises(planar: mujoco.MjModel) -> None:
    """A typo must not silently leave the joint movable: the two runs being
    compared would then differ by nothing, and the comparison would look done."""
    with pytest.raises(ValueError, match="no such joint"):
        mp.build_movable(planar, ["arm_joint3"])


def test_freezing_everything_raises(planar: mujoco.MjModel) -> None:
    with pytest.raises(ValueError, match="nothing to sample"):
        mp.build_movable(planar, [mp.YAW_LABEL, "j0", "j1"])


def test_zero_range_joint_is_refused() -> None:
    """MuJoCo compiles 0/0 as *unlimited*, so a raw CAD export would be sampled
    over a range of exactly nothing and reported as a workspace."""
    model = mujoco.MjModel.from_xml_string(UNLIMITED)
    mv = mp.build_movable(model, [])
    with pytest.raises(ValueError, match="no usable range"):
        mp.joint_limits(model, mv.arm_joints)


# --- the metric -----------------------------------------------------------


def test_w_is_zero_when_the_task_outranks_the_joints() -> None:
    """The singular-value trap.

    ``svd`` of a 6x5 matrix returns five values whose product is nonzero, so
    computing w that way reports a 5-DoF arm as capable of a 6D task. w is an
    m x m determinant and must vanish.
    """
    rng = np.random.default_rng(0)
    jac = rng.normal(size=(6, 5))
    w, full = mp.manipulability(jac)
    assert w == pytest.approx(0.0, abs=1e-9)
    assert not full
    # The trap itself, asserted so the two cannot be confused again.
    assert np.prod(np.linalg.svd(jac, compute_uv=False)) > 1e-3


def test_w_matches_the_singular_value_product_when_square() -> None:
    rng = np.random.default_rng(1)
    jac = rng.normal(size=(3, 5))
    w, full = mp.manipulability(jac)
    assert full
    assert w == pytest.approx(np.prod(np.linalg.svd(jac, compute_uv=False)))


def test_task_row_counts(spin: mujoco.MjModel) -> None:
    data = mujoco.MjData(spin)
    site = mujoco.mj_name2id(spin, mujoco.mjtObj.mjOBJ_SITE, "ee")
    mujoco.mj_kinematics(spin, data)
    mujoco.mj_comPos(spin, data)
    jacp = np.zeros((3, spin.nv))
    jacr = np.zeros((3, spin.nv))
    mujoco.mj_jacSite(spin, data, jacp, jacr, site)
    rows = {
        t: mp._task_rows(jacp, jacr, t, data.site_xmat[site]).shape[0]
        for t in ("position", "tool-axis", "pose")
    }
    assert rows == {"position": 3, "tool-axis": 5, "pose": 6}


def test_tool_axis_task_ignores_a_pure_spin_joint(spin: mujoco.MjModel) -> None:
    """A joint whose axis runs through the site along the tool direction adds
    nothing to position or to pointing -- only to spin, which this task drops.

    This is what makes freezing such a joint free for a tool-axis task and fatal
    for a 6D pose task, and it is the whole reason the two tasks are separate.
    """
    data = mujoco.MjData(spin)
    site = mujoco.mj_name2id(spin, mujoco.mjtObj.mjOBJ_SITE, "ee")
    data.qpos[spin.jnt_qposadr[1]] = 0.4
    data.qpos[spin.jnt_qposadr[2]] = -0.5
    mujoco.mj_kinematics(spin, data)
    mujoco.mj_comPos(spin, data)
    jacp = np.zeros((3, spin.nv))
    jacr = np.zeros((3, spin.nv))
    mujoco.mj_jacSite(spin, data, jacp, jacr, site)
    spin_dof = int(spin.jnt_dofadr[3])

    assert np.abs(jacp[:, spin_dof]).max() < 1e-12
    tool = data.site_xmat[site].reshape(3, 3)[:, 2]
    assert abs(abs(jacr[:, spin_dof] @ tool) - 1.0) < 1e-9

    jac5 = mp._task_rows(jacp, jacr, "tool-axis", data.site_xmat[site])
    assert np.abs(jac5[:, spin_dof]).max() < 1e-9


# --- the yaw shortcut -----------------------------------------------------
#
# sample() holds yaw at zero even when it is movable. These two tests are what
# make that complete rather than a slice: if either fails, the cloud is missing
# configurations and every fraction it reports is wrong.


def _probe(model: mujoco.MjModel, yaw: float, q: list[float], task: str):
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    mujoco.mj_resetData(model, data)
    data.qpos[3] = np.cos(yaw / 2.0)
    data.qpos[6] = np.sin(yaw / 2.0)
    mv = mp.build_movable(model, [])
    for j, val in zip(mv.arm_joints, q):
        data.qpos[model.jnt_qposadr[j]] = val
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    jac = mp._task_rows(jacp, jacr, task, data.site_xmat[site])[:, mv.dofs]
    w, _ = mp.manipulability(jac)
    return w, data.site_xpos[site].copy(), data.xpos[base].copy()


@pytest.mark.parametrize("task", ["position", "tool-axis", "pose"])
def test_w_is_invariant_to_yaw(spin: mujoco.MjModel, task: str) -> None:
    """Yaw rigidly rotates the whole chain, so it cannot change a singular value."""
    q = [0.4, -0.5, 0.9]
    w0, _, _ = _probe(spin, 0.0, q, task)
    for yaw in (0.9, -2.1, 3.0):
        w, _, _ = _probe(spin, yaw, q, task)
        assert w == pytest.approx(w0, rel=1e-9, abs=1e-15)


def test_base_frame_position_is_invariant_to_yaw(spin: mujoco.MjModel) -> None:
    """The other half: the reachable set *in the base frame* does not move
    either, so one yaw slice is the whole cloud."""
    q = [0.4, -0.5, 0.9]
    _, p0, origin = _probe(spin, 0.0, q, "pose")
    for yaw in (0.9, -2.1, 3.0):
        _, p, _ = _probe(spin, yaw, q, "pose")
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        back = rot @ (p - origin) + origin
        np.testing.assert_allclose(back, p0, atol=1e-12)


# --- sampling and reduction -----------------------------------------------


def test_planar_arm_is_deficient_for_a_3d_position_task(planar: mujoco.MjModel) -> None:
    """An analytic rank verdict: every hinge is about z, so the ee cannot leave
    its plane and no configuration spans 3D position."""
    mv = mp.build_movable(planar, [mp.YAW_LABEL])
    cloud = mp.sample(planar, "position", mv, 400, 0.04, 0, progress=False)
    assert cloud.deficient_fraction == 1.0
    assert not cloud.full_rank.any()
    assert np.allclose(cloud.w, 0.0, atol=1e-12)


def test_yaw_does_not_rescue_parallel_hinges(planar: mujoco.MjModel) -> None:
    """Yaw is about z and so is every hinge here, so yaw adds a column that is
    already spanned. Freeing it changes nothing -- an aerial manipulator does not
    get 3D position for free just because the vehicle can turn."""
    free = mp.sample(planar, "position", mp.build_movable(planar, []), 400, 0.04, 0,
                     progress=False)
    frozen = mp.sample(planar, "position", mp.build_movable(planar, [mp.YAW_LABEL]),
                       400, 0.04, 0, progress=False)
    assert free.deficient_fraction == 1.0
    assert frozen.deficient_fraction == 1.0


def test_yaw_rescues_a_non_parallel_arm(spin: mujoco.MjModel) -> None:
    """Hinges about y plus yaw about z do span 3D position, and without yaw they
    do not. So freezing yaw is a real change to the metric even though it leaves
    the cloud's shape alone -- which is why it is a --freeze target."""
    free = mp.sample(spin, "position", mp.build_movable(spin, []), 600, 0.04, 0,
                     progress=False)
    frozen = mp.sample(spin, "position", mp.build_movable(spin, [mp.YAW_LABEL]),
                       600, 0.04, 0, progress=False)
    assert free.deficient_fraction == 0.0
    assert frozen.deficient_fraction == 1.0
    # Same voxels either way: yaw only rotates the cloud, it does not extend it.
    assert len(free.pos) == len(frozen.pos)


def test_voxel_keeps_the_best_configuration(spin: mujoco.MjModel) -> None:
    """One voxel is reached many ways; the report is about whether the arm *can*
    work there, so the best sample wins and a full-rank sample is never masked
    by the deficient ones beside it."""
    mv = mp.build_movable(spin, [])
    coarse = mp.sample(spin, "position", mv, 800, 0.4, 0, progress=False)
    fine = mp.sample(spin, "position", mv, 800, 0.04, 0, progress=False)
    assert len(coarse.pos) < len(fine.pos)
    # Coarse voxels absorb more samples, so their best can only improve.
    assert coarse.w.max() >= fine.w.max() - 1e-12
    # A voxel counts as workable if *any* sample there met the rank, so a coarse
    # grid can never report more deficiency than a fine one.
    assert coarse.deficient_fraction <= fine.deficient_fraction + 1e-12


def test_cloud_is_in_the_base_frame(planar: mujoco.MjModel) -> None:
    """A cloud must not move when the vehicle is parked somewhere else, or two
    runs at different hover points would not be comparable."""
    mv = mp.build_movable(planar, [])
    first = mp.sample(planar, "position", mv, 300, 0.04, 7, progress=False)
    moved = mujoco.MjModel.from_xml_string(
        PLANAR.replace('pos="0 0 0.5"', 'pos="3 -2 1.25"')
    )
    second = mp.sample(moved, "position", mp.build_movable(moved, []), 300, 0.04, 7,
                       progress=False)
    np.testing.assert_allclose(first.pos, second.pos, atol=1e-12)


# --- drawing --------------------------------------------------------------


def _scene(model: mujoco.MjModel, cloud, **kw) -> int:
    scene = mujoco.MjvScene(model, maxgeom=mp.VIEWER_MAX_GEOM)
    scene.ngeom = 0
    return mp.draw(scene, cloud, np.zeros(3), 0.01, **kw)


def test_slice_keeps_a_slab_not_a_projection(planar: mujoco.MjModel) -> None:
    """A solid cloud renders only its shell, so the slab is the only honest way
    to see inside. It must drop voxels, not flatten them onto the plane.

    Sliced along y, which PLANAR actually spans -- SPIN_WRIST's hinges are all
    about y, so its whole cloud already lies in one y plane and a slice there
    would pass while doing nothing.
    """
    mv = mp.build_movable(planar, [])
    cloud = mp.sample(planar, "position", mv, 2000, 0.04, 0, progress=False)
    everything = _scene(planar, cloud)
    slab = _scene(planar, cloud, slice_axis="y", slice_at=0.0)
    assert 0 < slab < everything
    within = np.abs(cloud.pos[:, 1]) <= cloud.voxel / 2.0
    assert slab == int(within.sum())


def test_slice_width_widens_the_slab(planar: mujoco.MjModel) -> None:
    mv = mp.build_movable(planar, [])
    cloud = mp.sample(planar, "position", mv, 2000, 0.04, 0, progress=False)
    thin = _scene(planar, cloud, slice_axis="y", slice_at=0.0)
    thick = _scene(planar, cloud, slice_axis="y", slice_at=0.0, slice_width=0.3)
    assert thick > thin


def test_deficient_only_draws_nothing_when_nothing_is_deficient(
    spin: mujoco.MjModel,
) -> None:
    mv = mp.build_movable(spin, [])
    cloud = mp.sample(spin, "position", mv, 1000, 0.04, 0, progress=False)
    assert cloud.deficient_fraction == 0.0
    assert _scene(spin, cloud, deficient_only=True) == 0


def test_revolve_multiplies_the_cloud(spin: mujoco.MjModel) -> None:
    """Presentation only: w does not depend on yaw, so a revolved copy is the
    same measurement seen from another heading."""
    mv = mp.build_movable(spin, [])
    cloud = mp.sample(spin, "pose", mv, 1000, 0.06, 0, progress=False)
    one = _scene(spin, cloud)
    four = _scene(spin, cloud, revolve=4)
    assert four == 4 * one


def test_draw_stops_at_scene_capacity(spin: mujoco.MjModel) -> None:
    """Overflowing mjv_initGeom's buffer would be a segfault, not an exception."""
    mv = mp.build_movable(spin, [])
    cloud = mp.sample(spin, "pose", mv, 2000, 0.03, 0, progress=False)
    scene = mujoco.MjvScene(spin, maxgeom=40)
    scene.ngeom = 0
    drawn = mp.draw(scene, cloud, np.zeros(3), 0.01)
    assert drawn == 40
    assert scene.ngeom == 40


def test_shade_lightens_with_w() -> None:
    """The scene is dark, so light must mean high w -- the reverse of a chart on
    white. Monotone in lightness either way."""
    dim = mp._shade(0.0, 1.0)
    bright = mp._shade(1.0, 1.0)
    assert sum(bright[:3]) > sum(dim[:3])
    assert mp._shade(0.0, 0.0) == dim  # wmax == 0 must not divide by zero


def test_pose_and_tool_axis_agree_when_the_wrist_is_a_pure_spin(
    spin: mujoco.MjModel,
) -> None:
    """w is the same for a 6D and a 5D task here, which looks like a bug and is
    an identity.

    The spin joint's only contribution to the 6D Jacobian is the tool-axis row
    that the 5D task deletes. Expanding the determinant along that column leaves
    exactly the 5D one, so the two measures coincide -- but only while the joint
    is movable and its axis runs through the site.
    """
    data = mujoco.MjData(spin)
    site = mujoco.mj_name2id(spin, mujoco.mjtObj.mjOBJ_SITE, "ee")
    mv = mp.build_movable(spin, [])
    rng = np.random.default_rng(3)
    for _ in range(25):
        mujoco.mj_resetData(spin, data)
        for j in mv.arm_joints:
            lo, hi = spin.jnt_range[j]
            data.qpos[spin.jnt_qposadr[j]] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(spin, data)
        mujoco.mj_comPos(spin, data)
        jacp = np.zeros((3, spin.nv))
        jacr = np.zeros((3, spin.nv))
        mujoco.mj_jacSite(spin, data, jacp, jacr, site)
        xmat = data.site_xmat[site]
        w6, _ = mp.manipulability(mp._task_rows(jacp, jacr, "pose", xmat)[:, mv.dofs])
        w5, _ = mp.manipulability(
            mp._task_rows(jacp, jacr, "tool-axis", xmat)[:, mv.dofs]
        )
        # Loose rtol on purpose: w is a determinant of J J^T, so its error scales
        # with cond(J)**2, and near-singular poses lose most of the mantissa.
        assert w6 == pytest.approx(w5, rel=1e-5, abs=1e-14)

    # Freezing a pure-spin joint costs the position task nothing at all, which is
    # the other half of "its axis runs through the site". This chain has too few
    # DoF for a 5D task either way, so the tool-axis half of the contrast needs a
    # real arm -- it is measured on the X8 in the private study, not here.
    frozen = mp.build_movable(spin, ["spin"])
    free = mp.sample(spin, "position", mv, 600, 0.05, 0, progress=False)
    held = mp.sample(spin, "position", frozen, 600, 0.05, 0, progress=False)
    assert free.deficient_fraction == held.deficient_fraction == 0.0


def test_report_warns_that_w_is_not_comparable(planar: mujoco.MjModel) -> None:
    mv = mp.build_movable(planar, [mp.YAW_LABEL])
    text = mp.report(mp.sample(planar, "position", mv, 200, 0.04, 0, progress=False))
    assert "LARGER" in text
    assert "counting" in text  # the structural verdict, since this run is 100%
