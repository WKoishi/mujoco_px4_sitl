"""Tests for scripts/urdf_to_mjcf.py.

No CAD needed: every test builds a small URDF and a matching sidecar in a tmp
directory, so this runs in a fresh checkout with no private geometry present.

The focus is the failure modes that are **silent** in MuJoCo -- a welded base, a
rotor site on the wrong body, a gap in the rotor indices, joint limits that
compile to "unlimited", and CAD meshes deleted by ``discardvisual``. Each of
those produces a model that loads and flies, wrongly.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "urdf_to_mjcf.py"
_spec = importlib.util.spec_from_file_location("urdf_to_mjcf", _SCRIPT)
assert _spec is not None and _spec.loader is not None
u2m = importlib.util.module_from_spec(_spec)
sys.modules["urdf_to_mjcf"] = u2m
_spec.loader.exec_module(u2m)


# --- fixtures -------------------------------------------------------------


def _write_box_stl(path: Path, half: float = 0.1, subdivide: int = 1) -> int:
    """A closed box as binary STL.

    ``subdivide`` tessellates each face into an n x n grid, so the face count
    scales as ``12 * n^2`` -- enough to cross MuJoCo's decoder limit and exercise
    decimation without changing the shape.
    """
    corners = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64) * half
    quads = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
    ]
    triangles: list[np.ndarray] = []
    for a, b, c, d in quads:
        origin = corners[a]
        edge_u = corners[b] - origin
        edge_v = corners[d] - origin
        n = subdivide
        for i in range(n):
            for j in range(n):
                p00 = origin + edge_u * (i / n) + edge_v * (j / n)
                p10 = origin + edge_u * ((i + 1) / n) + edge_v * (j / n)
                p11 = origin + edge_u * ((i + 1) / n) + edge_v * ((j + 1) / n)
                p01 = origin + edge_u * (i / n) + edge_v * ((j + 1) / n)
                triangles.append(np.array([p00, p10, p11]))
                triangles.append(np.array([p00, p11, p01]))
    faces = np.array(triangles)
    # Via the script's own writer: vectorized, and it keeps the fixture honest
    # about the format the reader expects.
    flat = faces.reshape(-1, 3)
    u2m.write_stl(path, flat, np.arange(len(flat)).reshape(-1, 3))
    return len(faces)


_URDF = """<?xml version="1.0"?>
<robot name="rig">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.03"/>
    </inertial>
    <visual><geometry><mesh filename="package://rig/meshes/base_link.STL"/></geometry></visual>
    <collision><geometry><mesh filename="package://rig/meshes/base_link.STL"/></geometry></collision>
  </link>
  <link name="arm_link0">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual><geometry><mesh filename="package://rig/meshes/arm_link0.STL"/></geometry></visual>
  </link>
  <joint name="arm_joint0" type="revolute">
    <origin xyz="0.1 0 0.02" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="arm_link0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0" effort="0" velocity="0"/>
  </joint>
</robot>
"""


def _sidecar_dict(root: Path) -> dict:
    """A valid 4-rotor sidecar for the rig above."""
    return {
        "urdf": str(root / "urdf" / "rig.urdf"),
        "output": str(root / "out" / "rig.xml"),
        "rotors": [
            {"pos": [+0.2, -0.2, 0.0], "spin": "CCW"},
            {"pos": [-0.2, +0.2, 0.0], "spin": "CCW"},
            {"pos": [+0.2, +0.2, 0.0], "spin": "CW"},
            {"pos": [-0.2, -0.2, 0.0], "spin": "CW"},
        ],
        "joints": {
            "arm_joint0": {
                "lower": {"deg": -90}, "upper": {"deg": 90},
                "kp": 10.0, "kv": 0.5, "damping": 0.2, "armature": 0.01,
            },
        },
        "sites": {"imu": {"pos": [0.0, 0.0, 0.0]}},
        "collisions": [
            {"name": "hull", "type": "box", "pos": [0, 0, 0],
             "size": [0.12, 0.12, 0.03]},
        ],
        "world": {"floor": True, "spawn_height": 0.2},
        "expected_mass": 2.5,
    }


@pytest.fixture
def rig(tmp_path: Path):
    """A URDF + meshes + sidecar tree; returns ``(sidecar_path, sidecar_dict)``."""
    (tmp_path / "urdf").mkdir()
    (tmp_path / "meshes").mkdir()
    (tmp_path / "urdf" / "rig.urdf").write_text(_URDF)
    _write_box_stl(tmp_path / "meshes" / "base_link.STL", 0.12)
    _write_box_stl(tmp_path / "meshes" / "arm_link0.STL", 0.03)
    data = _sidecar_dict(tmp_path)
    path = tmp_path / "rig.conversion.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, data


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _convert(path: Path) -> tuple[mujoco.MjModel, u2m.CheckResult]:
    """Run the conversion the way main() does, and return model + checks."""
    sidecar = u2m.load_sidecar(path)
    mesh_dir = sidecar.output.parent / sidecar.mesh_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    _, mapping = u2m.prepare_meshes(
        sidecar, sidecar.urdf.read_text(), mesh_dir, write=True
    )
    spec = u2m.build_spec(sidecar, mesh_dir, mapping)
    model = spec.compile()
    return model, u2m.self_check(model, sidecar)


# --- the injected pieces --------------------------------------------------


def test_the_base_gets_a_freejoint_and_is_not_welded(rig):
    """MuJoCo's URDF parser welds the root link to the world.

    A welded base raises nothing -- it simply never accelerates, which reads as
    a thrust problem rather than a missing joint. Compare against the raw URDF
    to show the parser really does weld it.
    """
    path, _ = rig
    model, result = _convert(path)
    assert not result.failures, result.lines

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    first = int(model.body_jntadr[base])
    assert model.jnt_type[first] == mujoco.mjtJoint.mjJNT_FREE

    # Without the injection there is no freejoint at all.
    raw = mujoco.MjSpec.from_file(str(u2m.load_sidecar(path).urdf))
    assert not any(
        j.type == mujoco.mjtJoint.mjJNT_FREE for j in raw.body("base_link").joints
    )


def test_rotor_sites_are_contiguous_and_on_the_base(rig):
    """``vehicle.py`` scans ``rotorN`` and needs them all on ``base_link``."""
    path, _ = rig
    model, result = _convert(path)
    assert not result.failures, result.lines

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    for index in range(4):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"rotor{index}")
        assert site >= 0
        assert int(model.site_bodyid[site]) == base
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rotor4") < 0


def test_the_vehicle_that_comes_out_is_one_vehicle_py_accepts(rig):
    """The end-to-end point: the model drives the real rotor model.

    Hover at 0.450 and full-command thrust/weight of 4.0 are
    MODELING_CONVENTIONS.md 2.5's numbers, and both are mass-independent -- so
    this rig reproduces them despite being nothing like the X8.
    """
    from mujoco_px4_sitl.vehicle import RotorModel, Vehicle

    path, _ = rig
    model, _ = _convert(path)
    vehicle = Vehicle(model, RotorModel(spin=(+1, +1, -1, -1)))
    assert vehicle.num_rotors == 4
    assert vehicle.total_mass == pytest.approx(2.5, abs=1e-6)
    assert vehicle.hover_command() == pytest.approx(0.450, abs=0.001)

    vehicle.update_commands(np.full(4, vehicle.hover_command()), 1.0)
    force, _ = vehicle.wrench_body()
    assert force[2] == pytest.approx(vehicle.weight, rel=1e-9)


def test_imu_sensors_exist_and_are_three_axis(rig):
    path, _ = rig
    model, result = _convert(path)
    assert not result.failures, result.lines
    for name in ("imu_accel", "imu_gyro"):
        sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        assert sensor >= 0
        assert int(model.sensor_dim[sensor]) == 3


def test_cad_meshes_survive_being_taken_out_of_collision(rig):
    """``discardvisual`` defaults to true on the URDF path.

    It deletes every geom that does not collide -- which is exactly what the
    CAD meshes become, since their convex hull would otherwise wrap the vehicle
    (section 3.2). Left on, the model still compiles and flies with no visual
    geometry at all, and nothing in a headless run would show it.
    """
    path, _ = rig
    model, result = _convert(path)
    assert not result.failures, result.lines
    # Two distinct meshes; base_link references its one twice (the URDF gives
    # visual and collision the same file), so there are three mesh geoms.
    assert model.nmesh == 2

    mesh_geoms = [
        index for index in range(model.ngeom)
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    assert len(mesh_geoms) == 3
    # Every one of them is out of collision: that is what makes the model
    # landable, and what would have got them deleted by discardvisual.
    for index in mesh_geoms:
        assert int(model.geom_contype[index]) == 0
        assert int(model.geom_conaffinity[index]) == 0
    # arm_link0's geometry is the one the default would have dropped: its only
    # geom is a <visual>, with no <collision> sibling to save it.
    arm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_link0")
    assert int(model.body_geomnum[arm]) == 1


def test_the_world_gets_a_floor_because_a_urdf_cannot_describe_one(rig):
    """Without a floor the vehicle free-falls, and that reads as lost thrust."""
    path, _ = rig
    model, result = _convert(path)
    assert not result.failures, result.lines
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0

    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert data.ncon > 0
    assert data.qpos[2] > 0.0


def test_mass_and_inertia_come_from_the_urdf_not_from_geom_density(rig):
    """Added primitives must not contribute mass.

    ``<inertial>`` is SolidWorks' mass properties, the whole reason to route
    through CAD. A collision primitive with density would silently add to it,
    and mass is what sets the auto-calibrated ``c_t``.
    """
    path, data = rig
    model, _ = _convert(path)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert float(model.body_subtreemass[base]) == pytest.approx(2.5, abs=1e-9)

    data["collisions"].append(
        {"name": "extra", "type": "sphere", "pos": [0, 0, 0.2], "size": [0.05]}
    )
    model2, _ = _convert(_write(path, data))
    assert float(model2.body_subtreemass[base]) == pytest.approx(2.5, abs=1e-9)


# --- the sidecar's own validation ----------------------------------------


def test_zero_length_joint_limits_are_rejected(rig):
    """SolidWorks writes ``lower=upper=0``, which MuJoCo reads as *unlimited*.

    So the export's own default is not a locked joint but a free one, with a
    position actuator whose ctrlrange spans nothing. It has to be a hard error:
    the compiled model is otherwise indistinguishable from a configured one.
    """
    path, data = rig
    data["joints"]["arm_joint0"]["lower"] = 0.0
    data["joints"]["arm_joint0"]["upper"] = 0.0
    with pytest.raises(ValueError, match="never set in CAD"):
        u2m.load_sidecar(_write(path, data))


def test_that_zero_limits_really_do_compile_to_unlimited():
    """The premise of the test above, pinned against MuJoCo itself.

    If a future MuJoCo made 0/0 a locked joint instead, the error message would
    become misleading -- so the claim is verified rather than asserted.
    """
    spec = mujoco.MjSpec.from_string(_URDF.replace(
        'filename="package://rig/meshes/base_link.STL"', 'filename="none.STL"'
    ).replace("<mesh ", "<box size=\"0.1 0.1 0.1\" ").replace(
        'filename="none.STL"/>', "/>"
    ).replace('filename="package://rig/meshes/arm_link0.STL"/>', "/>"))
    model = spec.compile()
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint0")
    assert int(model.jnt_limited[joint]) == 0


def test_a_joint_in_the_urdf_but_not_the_sidecar_is_an_error(rig):
    path, data = rig
    data["joints"] = {}
    with pytest.raises(ValueError, match="arm_joint0"):
        _convert(_write(path, data))


def test_a_sidecar_joint_not_in_the_urdf_is_an_error(rig):
    path, data = rig
    data["joints"]["arm_joint9"] = dict(data["joints"]["arm_joint0"])
    with pytest.raises(ValueError, match="not in the URDF"):
        _convert(_write(path, data))


def test_a_misspelled_key_is_rejected_rather_than_ignored(rig):
    """A dropped ``omega_max`` leaves the placeholder in place and still flies."""
    path, data = rig
    data["rotors"][0]["omega_maxx"] = 900.0
    with pytest.raises(ValueError, match="unknown key"):
        u2m.load_sidecar(_write(path, data))


def test_c_t_without_omega_max_is_rejected(rig):
    """Only the product ``c_t * omega_max^2`` is physical.

    A measured `c_t` left against `vehicle.py`'s placeholder `omega_max` of
    1100 rad/s raises nothing and yields a plausible-looking model with wildly
    wrong thrust. The MN5008 fit is 8.4x the placeholder `c_t` precisely because
    its `omega_max` is 0.58x, so mixing the two gives thrust/weight 11.4 instead
    of 3.8.
    """
    path, data = rig
    data["rotors"][0]["c_t"] = 1.02e-04
    with pytest.raises(ValueError, match="c_t given without omega_max"):
        u2m.load_sidecar(_write(path, data))

    for rotor in data["rotors"]:
        rotor["c_t"] = 1.02e-04
        rotor["omega_max"] = 633.4
    sidecar = u2m.load_sidecar(_write(path, data))
    assert sidecar.rotors[0].omega_max == pytest.approx(633.4)


def test_measured_motor_numbers_give_the_reported_thrust_and_hover(rig):
    """The self-check's thrust/weight and hover must match ``vehicle.py``.

    `MPC_THR_HOVER` is set by `omega_idle / omega_max` and thrust-to-weight and
    nothing else, so the script's report is what a PX4 airframe gets tuned
    against -- it has to agree with the plant rather than approximate it.
    """
    from mujoco_px4_sitl.vehicle import RotorModel, Vehicle

    path, data = rig
    for rotor in data["rotors"]:
        rotor["c_t"] = 1.0209e-04
        rotor["omega_max"] = 633.4
        rotor["km"] = 0.02154
    data["omega_idle"] = 57.6
    path = _write(path, data)
    model, result = _convert(path)
    assert not result.failures, result.lines

    sidecar = u2m.load_sidecar(path)
    vehicle = Vehicle(model, RotorModel(
        c_t=1.0209e-04, omega_max=633.4, omega_idle=57.6, km=0.02154,
        spin=tuple(r.spin for r in sidecar.rotors),
    ))
    reported_hover = next(
        line for line in result.lines if "hover command" in line
    )
    assert f"{vehicle.hover_command():.4f}" in reported_hover

    vehicle.update_commands(np.ones(vehicle.num_rotors), 1.0)
    force, _ = vehicle.wrench_body()
    reported_ratio = next(line for line in result.lines if "thrust/weight" in line)
    assert f"{force[2] / vehicle.weight:.3f}" in reported_ratio

    # And hover really is where it says: thrust equals weight there.
    vehicle.update_commands(
        np.full(vehicle.num_rotors, vehicle.hover_command()), 1.0
    )
    hover_force, _ = vehicle.wrench_body()
    assert hover_force[2] == pytest.approx(vehicle.weight, rel=1e-6)


def test_a_vehicle_that_cannot_hover_fails_the_check(rig):
    """Under-powered is correct physics but presents as a broken model."""
    path, data = rig
    for rotor in data["rotors"]:
        rotor["c_t"] = 1.0e-06
        rotor["omega_max"] = 200.0
    data["omega_idle"] = 20.0
    _, result = _convert(_write(path, data))
    assert any("cannot hover" in f for f in result.failures), result.lines


def test_a_non_default_km_is_called_out_for_the_px4_airframe(rig):
    """PX4's `CA_ROTOR*_KM` defaults to 0.05; a real prop is rarely that.

    Left at the default the allocator expects the wrong yaw torque -- flyable,
    so it does not announce itself.
    """
    path, data = rig
    for rotor in data["rotors"]:
        rotor["km"] = 0.02154
        rotor["c_t"] = 1.0209e-04
        rotor["omega_max"] = 633.4
    table = u2m.rotor_table(u2m.load_sidecar(_write(path, data)))
    assert "CA_ROTOR*_KM must be set to 0.02154" in table
    assert "2.32x" in table


def test_spin_accepts_cw_ccw_and_rejects_anything_else(rig):
    path, data = rig
    data["rotors"][0]["spin"] = "CCW"
    assert u2m.load_sidecar(_write(path, data)).rotors[0].spin == +1
    data["rotors"][0]["spin"] = "CW"
    assert u2m.load_sidecar(_write(path, data)).rotors[0].spin == -1
    data["rotors"][0]["spin"] = "widdershins"
    with pytest.raises(ValueError, match="CW or CCW"):
        u2m.load_sidecar(_write(path, data))


def test_degrees_and_radians_are_both_accepted(rig):
    path, data = rig
    data["joints"]["arm_joint0"]["lower"] = {"deg": -45}
    data["joints"]["arm_joint0"]["upper"] = "90deg"
    sidecar = u2m.load_sidecar(_write(path, data))
    assert sidecar.joints["arm_joint0"].lower == pytest.approx(-np.pi / 4)
    assert sidecar.joints["arm_joint0"].upper == pytest.approx(np.pi / 2)


def test_a_missing_imu_site_is_rejected(rig):
    """``sim.py`` reads ``imu_accel`` / ``imu_gyro`` off it every frame."""
    path, data = rig
    data["sites"] = {}
    with pytest.raises(ValueError, match="sites.imu is required"):
        u2m.load_sidecar(_write(path, data))


def test_option_timestep_is_rejected_because_it_has_no_effect(rig):
    """``sim.py`` overwrites it from ``--physics-rate`` on every load."""
    path, data = rig
    data["option"] = {"timestep": 0.002}
    with pytest.raises(ValueError, match="no effect"):
        _convert(_write(path, data))


def test_a_mass_mismatch_against_cad_fails_the_check(rig):
    """A part with no material assigned is the usual cause, and mass sets c_t."""
    path, data = rig
    data["expected_mass"] = 3.4
    _, result = _convert(_write(path, data))
    assert any("expected_mass" in f for f in result.failures), result.lines


def test_a_rotor_count_mismatch_fails_the_check(rig):
    """The scan stops at the first gap, so a short list reads as a smaller
    vehicle with no error -- ``spin``'s length is the only other guard."""
    sidecar_path, data = rig
    sidecar = u2m.load_sidecar(sidecar_path)
    mesh_dir = sidecar.output.parent / sidecar.mesh_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    _, mapping = u2m.prepare_meshes(
        sidecar, sidecar.urdf.read_text(), mesh_dir, write=True
    )
    model = u2m.build_spec(sidecar, mesh_dir, mapping).compile()

    # Same model, but a sidecar claiming one more rotor than was injected.
    data["rotors"].append({"pos": [0.0, 0.3, 0.0], "spin": "CCW"})
    claims_five = u2m.load_sidecar(_write(sidecar_path, data))
    result = u2m.self_check(model, claims_five)
    assert any("contiguous" in f for f in result.failures), result.lines


def test_coaxial_pairs_with_matching_spins_are_flagged(rig):
    """Same-direction pair means no differential yaw, and hover looks fine."""
    path, data = rig
    data["rotors"] = [
        {"pos": [+0.2, +0.2, 0.0], "spin": "CCW", "deck": "upper"},
        {"pos": [-0.2, -0.2, 0.0], "spin": "CW", "deck": "upper"},
        {"pos": [+0.2, +0.2, -0.1], "spin": "CCW", "deck": "lower"},
        {"pos": [-0.2, -0.2, -0.1], "spin": "CW", "deck": "lower"},
    ]
    notes = u2m._describe_coaxial(u2m.load_sidecar(_write(path, data)))
    assert any("spins the same way" in n for n in notes), notes


# --- meshes ---------------------------------------------------------------


def test_meshes_above_mujocos_decoder_limit_are_decimated(tmp_path: Path):
    """Not a performance choice: MuJoCo refuses to load them at all.

    A raw SolidWorks export routinely exceeds 200k faces, so without this the
    model cannot compile, and the error names the STL rather than the cause.
    """
    (tmp_path / "urdf").mkdir()
    (tmp_path / "meshes").mkdir()
    (tmp_path / "urdf" / "rig.urdf").write_text(_URDF)
    # 12 * 130^2 faces: just over the decoder limit, which is the whole point.
    source = _write_box_stl(tmp_path / "meshes" / "base_link.STL", 0.12, subdivide=130)
    assert source > u2m.STL_FACE_LIMIT, source
    _write_box_stl(tmp_path / "meshes" / "arm_link0.STL", 0.03)

    data = _sidecar_dict(tmp_path)
    data["max_faces"] = 5000
    path = _write(tmp_path / "rig.conversion.yaml", data)

    sidecar = u2m.load_sidecar(path)
    mesh_dir = sidecar.output.parent / sidecar.mesh_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    reports, _ = u2m.prepare_meshes(
        sidecar, sidecar.urdf.read_text(), mesh_dir, write=True
    )
    base = next(r for r in reports if r.name == "base_link")
    assert base.source_faces == source
    assert base.output_faces <= 5000
    # Decimation is bounded by one grid cell, so the shape is preserved to a few
    # millimetres on a metre-scale body. These meshes are visual only.
    assert base.bbox_error < 0.01

    model, result = _convert(path)
    assert not result.failures, result.lines
    assert model.nmesh == 2


def test_an_ascii_stl_is_rejected_rather_than_misread(tmp_path: Path):
    path = tmp_path / "ascii.STL"
    path.write_text(
        "solid x\n facet normal 0 0 1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
        " endloop\n endfacet\nendsolid x\n"
    )
    with pytest.raises(ValueError, match="ASCII"):
        u2m.read_stl(path)


def test_stl_roundtrip_preserves_vertices(tmp_path: Path):
    _write_box_stl(tmp_path / "a.STL", 0.2)
    triangles = u2m.read_stl(tmp_path / "a.STL")
    flat = triangles.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    u2m.write_stl(tmp_path / "b.STL", unique, inverse.reshape(-1, 3))
    again = u2m.read_stl(tmp_path / "b.STL")
    assert again.shape == triangles.shape
    assert np.allclose(np.sort(again.reshape(-1, 3), axis=0),
                       np.sort(flat, axis=0), atol=1e-6)


def test_package_uris_are_rewritten_since_mujoco_cannot_read_them(rig):
    """MuJoCo has no notion of ``package://``; meshdir supplies the directory."""
    path, _ = rig
    sidecar = u2m.load_sidecar(path)
    mesh_dir = sidecar.output.parent / sidecar.mesh_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    _, mapping = u2m.prepare_meshes(
        sidecar, sidecar.urdf.read_text(), mesh_dir, write=True
    )
    assert mapping == {
        "package://rig/meshes/base_link.STL": "base_link.STL",
        "package://rig/meshes/arm_link0.STL": "arm_link0.STL",
    }


# --- the arm --------------------------------------------------------------


def test_arm_actuators_are_position_controlled_over_the_joint_range(rig):
    """``arm_cmd.values[i]`` is an absolute angle, so ctrlrange is the joint's."""
    path, _ = rig
    model, result = _convert(path)
    assert not result.failures, result.lines

    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_act0")
    assert actuator >= 0
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint0")
    assert int(model.jnt_limited[joint]) == 1
    assert np.allclose(model.actuator_ctrlrange[actuator], model.jnt_range[joint])
    assert np.allclose(model.jnt_range[joint], [-np.pi / 2, np.pi / 2])

    dof = int(model.jnt_dofadr[joint])
    assert model.dof_armature[dof] > 0.0
    assert model.dof_damping[dof] > 0.0


def test_the_arm_actually_tracks_a_commanded_angle(rig):
    """A position actuator that does not move is the point of the gains."""
    path, _ = rig
    model, _ = _convert(path)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint0")
    address = int(model.jnt_qposadr[joint])
    data.ctrl[0] = 0.5
    for _ in range(4000):
        mujoco.mj_step(model, data)
    assert data.qpos[address] == pytest.approx(0.5, abs=0.05)


def test_the_generated_file_reloads_from_disk(rig, tmp_path: Path):
    """What ``sim.py`` does. An in-memory spec working proves less."""
    path, _ = rig
    sidecar = u2m.load_sidecar(path)
    mesh_dir = sidecar.output.parent / sidecar.mesh_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    _, mapping = u2m.prepare_meshes(
        sidecar, sidecar.urdf.read_text(), mesh_dir, write=True
    )
    spec = u2m.build_spec(sidecar, mesh_dir, mapping)
    spec.compile()
    u2m.write_model(spec, sidecar, mesh_dir)

    assert sidecar.output.is_file()
    text = sidecar.output.read_text()
    assert 'meshdir="meshes"' in text
    assert "GENERATED by scripts/urdf_to_mjcf.py" in text

    model = mujoco.MjModel.from_xml_path(str(sidecar.output))
    assert model.nmesh == 2
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rotor3") >= 0


# --- the PX4 airframe ------------------------------------------------------
#
# Generated rather than hand-written because every PY is the negation of the
# model's py and KM's sign follows the spin, so transcription is where a
# mismatch gets in -- and a mismatch presents as yaw drift, which is routinely
# misdiagnosed as an EKF fault.


def test_airframe_py_is_the_negation_of_the_model_py(rig, tmp_path):
    """FLU -> FRD. Getting this wrong mirrors the vehicle left/right."""
    path, data = rig
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    text = out.read_text()
    # Sidecar rotor 0 is at y = -0.2, so the airframe must say +0.2.
    assert "CA_ROTOR0_PY 0.20000" in text
    assert "CA_ROTOR1_PY -0.20000" in text
    assert "CA_ROTOR0_PX 0.20000" in text


def test_airframe_km_sign_follows_spin(rig, tmp_path):
    path, _ = rig
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    text = out.read_text()
    assert "CA_ROTOR0_KM +0.05000" in text  # CCW
    assert "CA_ROTOR2_KM -0.05000" in text  # CW


def test_airframe_rotor_count_and_pwm_match_the_sidecar(rig, tmp_path):
    path, _ = rig
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    text = out.read_text()
    assert "CA_ROTOR_COUNT 4" in text
    assert "PWM_MAIN_FUNC4 104" in text
    assert "PWM_MAIN_FUNC5" not in text


def test_airframe_ct_is_thrust_at_full_command_not_c_t(rig, tmp_path):
    """CA_ROTOR*_CT is defined as Thrust = CT * u^2, so it is c_t * omega_max^2.

    Writing c_t here instead would be off by ~six orders of magnitude, and PX4
    would still boot.
    """
    path, data = rig
    for r in data["rotors"]:
        r["c_t"] = 1.0e-04
        r["omega_max"] = 600.0
    _write(path, data)
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    assert "CA_ROTOR0_CT 36.0000" in out.read_text()


def test_airframe_hover_matches_the_vehicle_the_simulator_builds(rig, tmp_path):
    """MPC_THR_HOVER must come from the same RotorModel sim.py will construct.

    Computing it any other way is how the airframe and the plant disagree while
    both stay flyable -- it presents as altitude oscillation.
    """
    from mujoco_px4_sitl.rotorconfig import load_rotors
    from mujoco_px4_sitl.vehicle import Vehicle

    path, data = rig
    for r in data["rotors"]:
        r["c_t"] = 1.0e-04
        r["omega_max"] = 600.0
    data["omega_idle"] = 55.0
    _write(path, data)
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)

    expected = Vehicle(model, load_rotors(path)).hover_command() ** 2
    assert f"MPC_THR_HOVER {expected:.4f}" in out.read_text()


def test_ct_factor_reaches_the_plant_and_not_ca_rotor_ct(rig, tmp_path):
    """The coaxial discount is a plant property. PX4 splits collective thrust in
    proportion to CA_ROTOR*_CT, so a discounted CT there would move PX4's hover
    point off the MPC_THR_HOVER derived here (the airframe template)."""
    from mujoco_px4_sitl.rotorconfig import load_rotors
    from mujoco_px4_sitl.vehicle import Vehicle

    path, data = rig
    for r in data["rotors"]:
        r["c_t"] = 1.0e-04
        r["omega_max"] = 600.0
        r["ct_factor"] = 0.8
    data["omega_idle"] = 55.0
    _write(path, data)
    model, result = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    text = out.read_text()

    assert "CA_ROTOR0_CT 36.0000" in text
    vehicle = Vehicle(model, load_rotors(path))
    assert vehicle.c_t[0] == pytest.approx(0.8e-04)
    assert f"MPC_THR_HOVER {vehicle.hover_command() ** 2:.4f}" in text
    reported = next(line for line in result.lines if "hover command" in line)
    assert f"{vehicle.hover_command():.4f}" in reported


def test_airframe_leaves_no_unfilled_placeholder(rig, tmp_path):
    path, _ = rig
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    assert "{{" not in out.read_text()


# --- attitude gains -------------------------------------------------------
#
# PX4's rate gains act on a normalized torque, so the airframe rescales them by
# the plant gain against the quad's (the airframe template says why). A flight
# shows only whether the result is right, not which piece is wrong, so each
# piece is pinned here: the reference, PX4's normalization, the inertia, and the
# hover acceleration behind the rate limit.

_QUAD_MODEL = Path(__file__).resolve().parents[1] / "models" / "quad_x.xml"


def _rig_vehicle(path: Path, model: mujoco.MjModel):
    sidecar = u2m.load_sidecar(path)
    return u2m.Vehicle(model, u2m.rotors_from_specs(sidecar.rotors, sidecar.omega_idle))


def test_the_quad_maps_onto_stock_px4():
    """Phase 5 flies the quad on stock gains, so the rule must leave it there:
    its rate loops already clear the separation on every axis, and its hover
    headroom clears every stock rate limit.
    """
    model = mujoco.MjModel.from_xml_path(str(_QUAD_MODEL))
    gains = u2m.attitude_gains(
        model, u2m.Vehicle(model), np.full(4, u2m.PX4_CA_ROTOR_CT_DEFAULT)
    )
    assert np.all(gains.separation >= u2m.RATE_LOOP_SEPARATION)
    np.testing.assert_array_equal(gains.rate_k, 1.0)
    np.testing.assert_array_equal(gains.rate_max_deg, u2m.PX4_RATE_MAX_DEG)


def test_gains_rise_only_as_far_as_the_separation_needs(rig, tmp_path):
    """A tenth of the quad's km slows the rig's yaw loop to about its attitude
    loop, so yaw K goes up to reach 4x; roll and pitch already clear it and stay
    at stock. Raising every axis to a reference vehicle's loop gain instead was
    flown on the X8 and sustained a roll limit cycle.
    """
    path, data = rig
    for r in data["rotors"]:
        r["km"] = 0.005
    _write(path, data)
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    text = out.read_text()

    gains = u2m.attitude_gains(model, _rig_vehicle(path, model), np.full(4, 6.5))
    yaw = u2m.RATE_LOOP_SEPARATION * 2.8 / (gains.plant_gain[2] * 0.2)
    assert 1.0 < yaw < u2m.PX4_RATE_K_MAX  # the case under test: raised, not capped
    assert f"MC_YAWRATE_K {yaw:.2f}" in text
    assert "MC_ROLLRATE_K 1.00" in text
    assert "MC_PITCHRATE_K 1.00" in text


def test_px4_normalization_gives_a_symmetric_xs_analytic_authority(rig):
    """For an X, PX4's scaling makes every roll/pitch entry 1/sqrt(2) and every
    yaw entry 1, so a unit command buys s*N*r/sqrt(2) and s*N*km.

    A wrong scale moves every derived K by the same factor, and nothing in a
    flight says why. The rig's arm puts its CoM off the rotor centre, which a
    torque column's zero net thrust must cancel.
    """
    path, _ = rig
    model, _ = _convert(path)
    vehicle = _rig_vehicle(path, model)
    gains = u2m.attitude_gains(model, vehicle, np.full(4, 6.5))

    hover = vehicle.hover_command()
    omega = vehicle.omega(np.full(4, hover))[0]
    slope = vehicle.c_t[0] * omega * (vehicle.omega_max[0] - vehicle.omega_idle) / hover
    np.testing.assert_allclose(gains.authority[:2], slope * 4 * 0.2 / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(gains.authority[2], slope * 4 * 0.05, rtol=1e-6)


def test_inertia_is_the_whole_vehicle_about_its_own_com(rig):
    """The rate loops turn the vehicle with the arm, about the composite CoM.

    By hand from the rig's URDF: base 2.0 kg at the origin, arm link 0.5 kg at
    (0.1, 0, 0.07), so the CoM is (0.02, 0, 0.014) and each body adds its
    parallel-axis term. Using base_link's own inertia instead is the mistake
    this rules out, and it would overstate every K by the arm's share.
    """
    path, _ = rig
    model, _ = _convert(path)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    com, inertia = u2m._subtree_mass_properties(model, base)
    np.testing.assert_allclose(com, [0.02, 0.0, 0.014], atol=1e-9)
    np.testing.assert_allclose(np.diag(inertia), [0.02296, 0.02696, 0.035], rtol=1e-6)
    assert inertia[0, 2] == pytest.approx(-0.0028, rel=1e-6)


def test_a_capped_gain_is_written_at_px4s_limit_and_reported(rig, tmp_path, capsys):
    """km 0.002 leaves yaw needing K ~ 10, past PX4's documented 5. It is written
    at 5 and said out loud, since the loop then sits short of the separation and
    a flight would show that without saying why.
    """
    path, data = rig
    for r in data["rotors"]:
        r["km"] = 0.002
    _write(path, data)
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)
    assert "MC_YAWRATE_K 5.00" in out.read_text()
    assert "clamped to PX4's 5" in capsys.readouterr().out


def test_yaw_rate_limit_is_hover_yaw_acceleration_over_the_attitude_gain(rig, tmp_path):
    """At hover the allocator holds thrust and backs yaw off once a motor reaches
    zero. For an X that is two rotors down to idle and two up to a = 2 * hover^2;
    the resulting torque over I_zz, divided by MC_YAW_P, is the fastest yaw the
    P law can still brake from.
    """
    path, data = rig
    for r in data["rotors"]:
        r["km"] = 0.005
    _write(path, data)
    model, _ = _convert(path)
    out = tmp_path / "22009_test_rig"
    u2m.emit_airframe(u2m.load_sidecar(path), model, out)

    vehicle = _rig_vehicle(path, model)
    hover = vehicle.hover_command()
    thrust = lambda u: float(vehicle.c_t[0] * vehicle.omega(np.array([u]))[0] ** 2)
    torque = 0.005 * 2 * (
        (thrust(hover) - thrust(0.0)) + (thrust(np.sqrt(2) * hover) - thrust(hover))
    )
    expected = np.degrees(torque / 0.035 / 2.8)
    assert expected < 200.0  # the case under test: the limit binds
    assert f"MC_YAWRATE_MAX {expected:.1f}" in out.read_text()


# --- propeller discs ------------------------------------------------------


def _with_radius(data: dict, radius: float | list[float | None]) -> dict:
    radii = radius if isinstance(radius, list) else [radius] * len(data["rotors"])
    for rotor, value in zip(data["rotors"], radii):
        if value is not None:
            rotor["radius"] = value
    return data


def test_a_rotor_radius_becomes_a_disc_shaped_site(rig):
    """What the simulator's clearance check reads. The site keeps its position,
    so thrust is unchanged; it only gains a shape."""
    path, data = rig
    model, result = _convert(_write(path, _with_radius(data, 0.1)))
    assert not result.failures, result.lines
    for index in range(4):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"rotor{index}")
        assert int(model.site_type[site]) == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert model.site_size[site, 0] == pytest.approx(0.1)
        assert model.site_pos[site] == pytest.approx(data["rotors"][index]["pos"])
    assert any("propeller discs, R = 0.1000 m" in line for line in result.lines)


def test_without_a_radius_the_arm_goes_unchecked_and_the_check_says_so(rig):
    path, _ = rig
    _, result = _convert(path)
    assert any("cannot detect the arm entering a propeller disc" in w for w in result.warnings)


def test_a_radius_on_some_rotors_only_fails(rig):
    path, data = rig
    _, result = _convert(_write(path, _with_radius(data, [0.1, 0.1, None, 0.1])))
    assert any("radius given for some rotors but not [2]" in f for f in result.failures)


def test_a_diameter_given_as_a_radius_fails_as_overlapping_blades(rig):
    """Hubs 0.4 m apart cannot carry 0.25 m blades; the overlap is the tell."""
    path, data = rig
    _, result = _convert(_write(path, _with_radius(data, 0.25)))
    assert any("blades would strike each other" in f for f in result.failures)


def test_a_zero_radius_is_rejected(rig):
    path, data = rig
    with pytest.raises(ValueError, match="radius must be > 0"):
        _convert(_write(path, _with_radius(data, 0.0)))


def test_the_clearance_survey_reports_how_much_of_the_envelope_reaches_a_disc(
    rig, monkeypatch
):
    """The rig's arm yaws a capsule 10 mm below the disc plane, so every pose
    that swings it over a disc is an intrusion 10 mm deep: capsule radius 20 mm
    minus the gap. Uses the simulator's own check, so the number is the one a
    flight would report."""
    monkeypatch.setattr(u2m, "CLEARANCE_SURVEY_POSES", 400)
    path, data = rig
    data = _with_radius(data, 0.1)
    data["collisions"].append({
        "name": "arm_link0_col", "type": "capsule", "body": "arm_link0",
        "fromto": [0.0, 0.0, -0.01, 0.2, 0.0, -0.01], "size": [0.02],
    })
    _, result = _convert(_write(path, data))
    survey = [w for w in result.warnings if "poses inside the joint limits" in w]
    assert len(survey) == 1, result.lines
    assert "deepest 10.0 mm (arm_link0_col in rotor" in survey[0]
