#!/usr/bin/env python3
"""URDF -> MJCF conversion for the aerial manipulator (MODELING_CONVENTIONS.md 6).

The CAD is revised repeatedly, so conversion is a script rather than a one-off
hand edit::

    your.urdf  +  <name>.conversion.yaml  ->  <name>.xml  (+ decimated meshes)

The URDF is only a courier for geometry, inertia and the kinematic tree
(section 1). Everything URDF cannot express lives in the sidecar YAML, so
re-exporting from CAD means re-running this script and losing no hand edits.

What is injected, and why URDF cannot carry it:

==========================  ====================================================
``freejoint``               MuJoCo's URDF parser welds the root link to the
                            world -- without this the vehicle cannot move at all
``rotor0..N`` sites          thrust application points; ``vehicle.py`` scans
                            these names and applies ``xfrc_applied`` at them
``imu`` site                 IMU mount, hardcoded name and ``quat="1 0 0 0"``
``imu_accel`` / ``imu_gyro`` the sensors ``sim.py`` reads every frame
``ee`` site                  optional end-effector reference (section 8)
``<position>`` actuators     ``arm_act0..`` driving the arm joints
joint ranges                 SolidWorks exports ``lower=upper=0``, which MuJoCo
                            compiles as *unlimited* -- not locked
``damping`` / ``armature``   what keeps a stiff position actuator stable

Run with ``--check-only`` to validate an existing conversion without writing.

Usage::

    python scripts/urdf_to_mjcf.py <sidecar>.conversion.yaml
    python scripts/urdf_to_mjcf.py <sidecar>.conversion.yaml --check-only
"""

from __future__ import annotations

import argparse
import logging
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from numpy.typing import NDArray

_log = logging.getLogger("urdf_to_mjcf")

# MuJoCo's binary STL decoder refuses a file above this face count, so a raw
# SolidWorks export (routinely 300k+ faces) cannot be loaded at all. Decimation
# is a hard requirement, not a performance preference -- see _decimate.
STL_FACE_LIMIT = 200_000

# Names the simulator hardcodes. Changing one here changes nothing in src/, so
# they are asserted rather than configured (MODELING_CONVENTIONS.md 3.3).
BASE_BODY = "base_link"
IMU_SITE = "imu"
ROTOR_PREFIX = "rotor"
ARM_JOINT_PREFIX = "arm_joint"
ARM_ACTUATOR_PREFIX = "arm_act"


# --- sidecar schema -------------------------------------------------------
#
# Dataclasses rather than raw dict access so a typo in the YAML is a named
# error at load time instead of a silently missing rotor.


@dataclass
class RotorSpec:
    """One rotor: where thrust is applied, and its coefficients.

    ``pos`` is FLU, in ``base_link``'s frame, and the *index is PX4's* --
    ``HIL_ACTUATOR_CONTROLS.controls[i]`` drives ``rotor{i}``, so this list's
    order must match ``CA_ROTOR*`` in the airframe file up to the FLU/FRD
    y-sign. Getting it wrong presents as yaw drift or attitude cross-coupling
    and is routinely misdiagnosed as an EKF fault.

    ``pz`` (the site's z) is documentation only: thrust is along body +z, so
    ``r x [0,0,T]`` uses only x and y. A coaxial deck's height contributes no
    torque, in our model and in PX4's allocator alike (section 4).
    """

    pos: tuple[float, float, float]
    spin: int
    c_t: float | None = None
    km: float = 0.05
    omega_max: float | None = None
    omega_min: float | None = None
    zaxis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    deck: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if self.spin not in (1, -1):
            raise ValueError(
                f"rotor {self.label or '?'}: spin must be +1 (CCW about body "
                f"+z) or -1 (CW), got {self.spin!r}"
            )
        if len(self.pos) != 3:
            raise ValueError(f"rotor {self.label or '?'}: pos needs 3 numbers")
        if self.c_t is not None and self.omega_max is None:
            # Thrust is c_t * omega^2, so only the product means anything. A
            # measured c_t left against vehicle.py's placeholder omega_max of
            # 1100 rad/s raises nothing and produces a plausible-looking model
            # with wildly wrong thrust -- the MN5008 fit came out 8.4x the
            # placeholder c_t precisely because its omega_max is 0.58x, and
            # mixing the two gives thrust/weight 11.4 instead of 3.8.
            raise ValueError(
                f"rotor {self.label or '?'}: c_t given without omega_max. "
                f"Thrust is c_t * omega^2, so a measured c_t against the "
                f"placeholder omega_max silently rescales every thrust in the "
                f"model. Give both, from the same datasheet"
            )


@dataclass
class JointSpec:
    """One arm joint: limits, actuator gains, and the dynamics that stabilize it.

    ``lower``/``upper`` are radians and **must be given** -- SolidWorks writes
    ``lower=upper=0`` when the limits were never set in CAD, and MuJoCo reads
    that as *unlimited*, so a missing limit is not a locked joint but a free
    one, with a position actuator whose ``ctrlrange`` is meaningless.

    ``damping`` and ``armature`` are gearbox friction and reflected inertia.
    They are not cosmetic: a stiff position actuator on an undamped, inertialess
    hinge oscillates at the physics rate.
    """

    lower: float
    upper: float
    kp: float
    damping: float
    armature: float
    kv: float = 0.0
    frictionloss: float = 0.0
    force_range: tuple[float, float] | None = None
    gear_ratio: float | None = None
    home: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            raise ValueError(
                f"{self.name}: upper ({self.upper}) must exceed lower "
                f"({self.lower}). SolidWorks exports 0/0 when the limits were "
                f"never set in CAD, and MuJoCo reads that as unlimited"
            )
        if not self.lower <= self.home <= self.upper:
            raise ValueError(
                f"{self.name}: home ({self.home}) is outside "
                f"[{self.lower}, {self.upper}]"
            )
        if self.kp <= 0.0:
            raise ValueError(f"{self.name}: kp must be > 0")


@dataclass
class SiteSpec:
    """A named site that is not a rotor: ``imu``, ``ee``, or a marker."""

    pos: tuple[float, float, float]
    body: str = BASE_BODY
    quat: tuple[float, float, float, float] | None = None


@dataclass
class CollisionSpec:
    """A hand-written collision primitive replacing a CAD mesh (section 3.2).

    The convex hull of 8 propellers wraps the whole vehicle, so mesh collision
    makes takeoff, landing and arm motion all hit the ground. Visual keeps the
    CAD mesh; collision uses these.
    """

    name: str
    type: str
    body: str = BASE_BODY
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, ...] = ()
    fromto: tuple[float, ...] | None = None
    quat: tuple[float, float, float, float] | None = None


@dataclass
class WorldSpec:
    """The scene around the vehicle, which a URDF cannot describe at all.

    Without a floor the vehicle has nothing to take off from or land on -- it
    free-falls forever, and the drop reads as a thrust problem rather than a
    missing scene. ``quad_x.xml`` carries the same pieces; keeping them here
    means both models present the same world to PX4.
    """

    floor: bool = True
    floor_size: float = 60.0
    spawn_height: float | None = None
    light: bool = True
    markers: bool = True


@dataclass
class Sidecar:
    """The whole hand-maintained configuration (MODELING_CONVENTIONS.md 6)."""

    urdf: Path
    output: Path
    rotors: list[RotorSpec]
    joints: dict[str, JointSpec]
    sites: dict[str, SiteSpec]
    world: WorldSpec = field(default_factory=WorldSpec)
    collisions: list[CollisionSpec] = field(default_factory=list)
    mesh_dir: str = "meshes"
    mesh_scale: float = 1.0
    max_faces: int = 60_000
    decimate: dict[str, int] = field(default_factory=dict)
    collision_meshes_off: bool = True
    omega_idle: float | None = None
    option: dict[str, Any] = field(default_factory=dict)
    expected_mass: float | None = None
    expected_com: tuple[float, float, float] | None = None
    mass_tolerance: float = 0.02
    notes: str = ""


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _angle(value: Any, where: str) -> float:
    """Radians from a number, or from ``{deg: X}`` / a ``"Xdeg"`` string.

    CAD limits are read off in degrees, and hand-converting them in the YAML is
    exactly where a sign or a factor gets lost.
    """
    if isinstance(value, dict):
        if "deg" in value:
            return math.radians(float(value["deg"]))
        if "rad" in value:
            return float(value["rad"])
        raise ValueError(f"{where}: angle mapping needs 'deg' or 'rad'")
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("deg"):
            return math.radians(float(text[:-3]))
        if text.endswith("rad"):
            return float(text[:-3])
        return float(text)
    return float(value)


def _unknown_keys(mapping: dict, allowed: set[str], where: str) -> None:
    """Reject unrecognized keys instead of ignoring them.

    A misspelled ``omega_max`` that is silently dropped leaves the placeholder
    in place and flies -- badly, and with no indication why.
    """
    extra = set(mapping) - allowed
    if extra:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}"
        )


def load_sidecar(path: Path) -> Sidecar:
    """Parse and validate the sidecar YAML. Raises on anything ambiguous."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    _unknown_keys(
        raw,
        {
            "urdf", "output", "rotors", "joints", "sites", "collisions",
            "world", "mesh_dir", "mesh_scale", "max_faces", "decimate",
            "collision_meshes_off", "omega_idle", "option", "expected_mass",
            "expected_com", "mass_tolerance", "notes",
        },
        str(path),
    )
    base = path.parent

    def resolve(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    rotors: list[RotorSpec] = []
    for index, entry in enumerate(_require(raw, "rotors", str(path))):
        where = f"{path}: rotors[{index}]"
        _unknown_keys(
            entry,
            {"pos", "spin", "c_t", "km", "omega_max", "omega_min", "zaxis",
             "deck", "label"},
            where,
        )
        spin_raw = _require(entry, "spin", where)
        if isinstance(spin_raw, str):
            text = spin_raw.strip().upper()
            if text not in ("CW", "CCW"):
                raise ValueError(f"{where}: spin string must be CW or CCW")
            spin = 1 if text == "CCW" else -1
        else:
            spin = int(spin_raw)
        rotors.append(RotorSpec(
            pos=tuple(float(v) for v in _require(entry, "pos", where)),
            spin=spin,
            c_t=entry.get("c_t"),
            km=float(entry.get("km", 0.05)),
            omega_max=entry.get("omega_max"),
            omega_min=entry.get("omega_min"),
            zaxis=tuple(float(v) for v in entry.get("zaxis", (0.0, 0.0, 1.0))),
            deck=str(entry.get("deck", "")),
            label=str(entry.get("label", f"{ROTOR_PREFIX}{index}")),
        ))
    if not rotors:
        raise ValueError(f"{path}: 'rotors' is empty")

    joints: dict[str, JointSpec] = {}
    for name, entry in (raw.get("joints") or {}).items():
        where = f"{path}: joints.{name}"
        _unknown_keys(
            entry,
            {"lower", "upper", "kp", "kv", "damping", "armature",
             "frictionloss", "force_range", "gear_ratio", "home"},
            where,
        )
        force_range = entry.get("force_range")
        joints[name] = JointSpec(
            lower=_angle(_require(entry, "lower", where), where),
            upper=_angle(_require(entry, "upper", where), where),
            kp=float(_require(entry, "kp", where)),
            kv=float(entry.get("kv", 0.0)),
            damping=float(_require(entry, "damping", where)),
            armature=float(_require(entry, "armature", where)),
            frictionloss=float(entry.get("frictionloss", 0.0)),
            force_range=(
                tuple(float(v) for v in force_range) if force_range else None
            ),
            gear_ratio=(
                float(entry["gear_ratio"]) if entry.get("gear_ratio") else None
            ),
            home=_angle(entry.get("home", 0.0), where),
            name=name,
        )

    sites: dict[str, SiteSpec] = {}
    for name, entry in (raw.get("sites") or {}).items():
        where = f"{path}: sites.{name}"
        _unknown_keys(entry, {"pos", "body", "quat"}, where)
        quat = entry.get("quat")
        sites[name] = SiteSpec(
            pos=tuple(float(v) for v in _require(entry, "pos", where)),
            body=str(entry.get("body", BASE_BODY)),
            quat=tuple(float(v) for v in quat) if quat else None,
        )
    if IMU_SITE not in sites:
        raise ValueError(
            f"{path}: sites.{IMU_SITE} is required -- sim.py reads imu_accel / "
            f"imu_gyro off it every frame"
        )

    collisions: list[CollisionSpec] = []
    for index, entry in enumerate(raw.get("collisions") or []):
        where = f"{path}: collisions[{index}]"
        _unknown_keys(
            entry, {"name", "type", "body", "pos", "size", "fromto", "quat"}, where
        )
        fromto = entry.get("fromto")
        quat = entry.get("quat")
        collisions.append(CollisionSpec(
            name=str(_require(entry, "name", where)),
            type=str(_require(entry, "type", where)),
            body=str(entry.get("body", BASE_BODY)),
            pos=tuple(float(v) for v in entry.get("pos", (0.0, 0.0, 0.0))),
            size=tuple(float(v) for v in entry.get("size", ())),
            fromto=tuple(float(v) for v in fromto) if fromto else None,
            quat=tuple(float(v) for v in quat) if quat else None,
        ))

    world_raw = raw.get("world") or {}
    _unknown_keys(
        world_raw,
        {"floor", "floor_size", "spawn_height", "light", "markers"},
        f"{path}: world",
    )
    world = WorldSpec(
        floor=bool(world_raw.get("floor", True)),
        floor_size=float(world_raw.get("floor_size", 60.0)),
        spawn_height=(
            float(world_raw["spawn_height"])
            if world_raw.get("spawn_height") is not None else None
        ),
        light=bool(world_raw.get("light", True)),
        markers=bool(world_raw.get("markers", True)),
    )

    return Sidecar(
        urdf=resolve(str(_require(raw, "urdf", str(path)))),
        output=resolve(str(_require(raw, "output", str(path)))),
        rotors=rotors,
        joints=joints,
        sites=sites,
        world=world,
        collisions=collisions,
        mesh_dir=str(raw.get("mesh_dir", "meshes")),
        mesh_scale=float(raw.get("mesh_scale", 1.0)),
        max_faces=int(raw.get("max_faces", 60_000)),
        decimate={str(k): int(v) for k, v in (raw.get("decimate") or {}).items()},
        collision_meshes_off=bool(raw.get("collision_meshes_off", True)),
        omega_idle=raw.get("omega_idle"),
        option=dict(raw.get("option") or {}),
        expected_mass=raw.get("expected_mass"),
        expected_com=(
            tuple(float(v) for v in raw["expected_com"])
            if raw.get("expected_com") else None
        ),
        mass_tolerance=float(raw.get("mass_tolerance", 0.02)),
        notes=str(raw.get("notes", "")),
    )


# --- meshes ---------------------------------------------------------------


def read_stl(path: Path) -> NDArray[np.float64]:
    """Triangles from a binary STL, shape ``(n, 3, 3)``.

    ASCII STL is rejected rather than parsed: SolidWorks writes binary, and an
    ASCII file here means the export settings were not the ones assumed.
    """
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: too short to be a binary STL")
    if data[:5].lstrip()[:5].lower() == b"solid" and b"facet normal" in data[:2048]:
        raise ValueError(
            f"{path}: looks like ASCII STL; re-export as binary (MuJoCo's "
            f"decoder needs binary, and the face count check relies on it)"
        )
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + count * 50
    if len(data) < expected:
        raise ValueError(
            f"{path}: header says {count} faces but the file holds "
            f"{(len(data) - 84) // 50}"
        )
    block = np.frombuffer(data[84:expected], dtype=np.uint8).reshape(count, 50)
    # Bytes 12..48 of each 50-byte record are the three vertices; 0..12 is the
    # normal, which is recomputed on write, and 48..50 the attribute count.
    return block[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)


def write_stl(path: Path, vertices: NDArray[np.float64], faces: NDArray[np.int64]) -> None:
    """Write a binary STL, recomputing facet normals from winding."""
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    length[length == 0.0] = 1.0
    normal = normal / length
    out = bytearray(b"\0" * 80)
    out += struct.pack("<I", len(faces))
    record = np.zeros((len(faces), 50), dtype=np.uint8)
    record[:, 0:12] = normal.astype("<f4").view(np.uint8).reshape(-1, 12)
    record[:, 12:48] = tri.astype("<f4").view(np.uint8).reshape(-1, 36)
    out += record.tobytes()
    path.write_bytes(bytes(out))


def _decimate(
    tri: NDArray[np.float64], target_faces: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Vertex-clustering decimation onto a uniform grid.

    Chosen over edge-collapse because it needs no extra dependency (the repo
    pins mujoco / pymavlink / numpy and nothing else) and it is bounded: the
    error is at most one cell diagonal, so a grid sized to the body gives a
    visual mesh accurate to a few millimetres on a metre-scale airframe. These
    meshes are visual only -- collision uses the hand-written primitives of
    section 3.2 -- so shape fidelity matters and watertightness does not.

    Not a preference but a requirement: MuJoCo's STL decoder rejects anything
    above STL_FACE_LIMIT faces, and SolidWorks exports routinely exceed it.
    """
    points = tri.reshape(-1, 3)
    span = float((points.max(axis=0) - points.min(axis=0)).max())
    if span <= 0.0:
        raise ValueError("degenerate mesh: zero extent")

    # Bisect on the grid resolution. Face count rises monotonically with it, so
    # bracket first -- coarsening until something fits, doubling until something
    # does not -- then narrow. A single scaled guess is not enough: the count
    # depends on how the surface happens to fall across the grid, so an estimate
    # can overshoot the target and must be allowed to come back down.
    low, high = 4, 64
    best: tuple[NDArray[np.float64], NDArray[np.int64]] | None = None

    while True:
        vertices, faces = _cluster(points, span, high)
        if len(faces) > target_faces:
            break
        best = (vertices, faces)
        if high >= 4096:
            return best
        low, high = high, high * 2

    while low < high - 1:
        middle = (low + high) // 2
        vertices, faces = _cluster(points, span, middle)
        if len(faces) <= target_faces:
            best = (vertices, faces)
            low = middle
        else:
            high = middle

    if best is None:
        # Even the coarsest grid tried overshoots. Keep halving rather than
        # returning something over budget: above STL_FACE_LIMIT MuJoCo refuses
        # the file outright, so an over-budget mesh is not a usable fallback.
        cells = low
        while cells > 1:
            cells = max(1, cells // 2)
            vertices, faces = _cluster(points, span, cells)
            if len(faces) <= target_faces:
                return vertices, faces
        raise ValueError(
            f"cannot decimate below {target_faces} faces; raise max_faces"
        )
    return best


def _cluster(
    points: NDArray[np.float64], span: float, cells: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Collapse vertices into a ``cells``-per-longest-axis grid."""
    origin = points.min(axis=0)
    step = span / cells
    index = np.floor((points - origin) / step).astype(np.int64)
    # Pack the 3 cell coordinates into one key. 21 bits each is ample: cells
    # stays well under 2^21 for any grid this routine will try.
    key = (index[:, 0] * (1 << 21) + index[:, 1]) * (1 << 21) + index[:, 2]
    unique, inverse = np.unique(key, return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique))
    representative = np.empty((len(unique), 3))
    for axis in range(3):
        representative[:, axis] = (
            np.bincount(inverse, weights=points[:, axis], minlength=len(unique))
            / count
        )
    faces = inverse.reshape(-1, 3)
    # Drop triangles whose corners collapsed together, then duplicates.
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[keep]
    if len(faces) == 0:
        raise ValueError("decimation collapsed every triangle; grid too coarse")
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    used, remapped = np.unique(faces, return_inverse=True)
    return representative[used], remapped.reshape(-1, 3).astype(np.int64)


@dataclass
class MeshReport:
    name: str
    source_faces: int
    output_faces: int
    bbox_error: float


def prepare_meshes(
    sidecar: Sidecar, urdf_text: str, out_dir: Path, write: bool
) -> tuple[list[MeshReport], dict[str, str]]:
    """Decimate every mesh the URDF references; return reports and a rename map.

    ``package://pkg/meshes/x.STL`` is rewritten to a bare filename, because
    MuJoCo does not understand package URIs and ``<compiler meshdir>`` supplies
    the directory (section 3.5).
    """
    source_dir = sidecar.urdf.parent
    reports: list[MeshReport] = []
    mapping: dict[str, str] = {}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)

    for reference in sorted(set(_mesh_references(urdf_text))):
        source = _resolve_mesh(reference, source_dir)
        stem = source.name
        mapping[reference] = stem
        triangles = read_stl(source)
        target = sidecar.decimate.get(source.stem, sidecar.max_faces)
        if len(triangles) <= target:
            vertices = triangles.reshape(-1, 3)
            unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
            faces = inverse.reshape(-1, 3).astype(np.int64)
            reduced = (unique, faces)
        else:
            reduced = _decimate(triangles, target)
        vertices, faces = reduced
        if len(faces) > STL_FACE_LIMIT:
            raise ValueError(
                f"{source.name}: {len(faces)} faces exceeds MuJoCo's STL "
                f"decoder limit of {STL_FACE_LIMIT}; lower max_faces"
            )
        original = triangles.reshape(-1, 3)
        error = float(
            np.abs(vertices.min(axis=0) - original.min(axis=0)).max()
            if len(vertices) else 0.0
        )
        error = max(error, float(
            np.abs(vertices.max(axis=0) - original.max(axis=0)).max()
        ))
        reports.append(MeshReport(source.stem, len(triangles), len(faces), error))
        if write:
            write_stl(out_dir / stem, vertices, faces)
    return reports, mapping


def _mesh_references(urdf_text: str) -> list[str]:
    """Every ``filename="..."`` in the URDF, in document order."""
    found: list[str] = []
    cursor = 0
    while True:
        at = urdf_text.find("filename=", cursor)
        if at < 0:
            return found
        quote = urdf_text[at + 9]
        end = urdf_text.index(quote, at + 10)
        found.append(urdf_text[at + 10:end])
        cursor = end


def _resolve_mesh(reference: str, urdf_dir: Path) -> Path:
    """Locate a mesh named by a ``package://`` URI or a relative path."""
    if reference.startswith("package://"):
        tail = reference[len("package://"):]
        _, _, relative = tail.partition("/")
    else:
        relative = reference
    # The URDF sits in <export>/urdf/, meshes in <export>/meshes/, so the
    # package-relative path resolves from the export root, one level up.
    for candidate in (
        urdf_dir / relative,
        urdf_dir.parent / relative,
        urdf_dir.parent / "meshes" / Path(relative).name,
        urdf_dir / Path(relative).name,
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"mesh {reference!r} not found near {urdf_dir} (tried package-relative "
        f"and meshes/)"
    )


# --- injection ------------------------------------------------------------


def build_spec(sidecar: Sidecar, mesh_dir: Path, mapping: dict[str, str]) -> mujoco.MjSpec:
    """Load the URDF and inject everything URDF cannot express."""
    text = sidecar.urdf.read_text(encoding="utf-8")
    for reference, stem in mapping.items():
        text = text.replace(f'"{reference}"', f'"{stem}"')
    text = _disable_discardvisual(text)

    # MjSpec resolves mesh paths against the file's own directory, so the URDF
    # is parsed from a copy sitting next to the decimated meshes.
    staged = mesh_dir / "_staged.urdf"
    staged.write_text(text, encoding="utf-8")
    try:
        spec = mujoco.MjSpec.from_file(str(staged))
    finally:
        staged.unlink(missing_ok=True)

    spec.modelname = sidecar.output.stem
    if spec.compiler.discardvisual:
        # Belt and braces: _disable_discardvisual should already have handled it
        # in the staged text, where it has to happen (see that function).
        spec.compiler.discardvisual = False
    # meshdir stays empty here: the staged URDF sits in the mesh directory, so
    # bare filenames resolve at compile time. The emitted XML sits one level up
    # and needs <compiler meshdir="..."> instead -- set in write_model, after
    # compiling, since changing it earlier breaks mesh lookup.
    if sidecar.mesh_scale != 1.0:
        for mesh in spec.meshes:
            mesh.scale = [sidecar.mesh_scale] * 3

    base = spec.body(BASE_BODY)
    if base is None:
        raise ValueError(
            f"{sidecar.urdf}: no body named {BASE_BODY!r}. The name is hardcoded "
            f"in vehicle.py and cannot be configured"
        )

    _inject_freejoint(spec, base)
    _name_geoms(spec, sidecar)
    _inject_sites(spec, sidecar)
    _inject_sensors(spec)
    _inject_arm(spec, sidecar)
    _inject_collisions(spec, sidecar)
    _inject_world(spec, sidecar, base)
    _apply_option(spec, sidecar)
    return spec


def _inject_world(spec: mujoco.MjSpec, sidecar: Sidecar, base: Any) -> None:
    """Add the floor, a light, and the ENU orientation markers.

    A URDF describes one robot and has no syntax for a scene, so the ground
    plane has to come from here. Without it the vehicle free-falls and the drop
    looks like a thrust deficit instead of a missing floor.
    """
    world = sidecar.world
    if world.spawn_height is not None:
        base.pos = [float(base.pos[0]), float(base.pos[1]), world.spawn_height]

    if world.light:
        light = spec.worldbody.add_light()
        light.pos = [0.0, 0.0, 6.0]
        light.dir = [0.0, 0.0, -1.0]
        light.diffuse = [0.8, 0.8, 0.8]

    if world.floor:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [world.floor_size, world.floor_size, 0.1]
        floor.pos = [0.0, 0.0, 0.0]
        floor.rgba = [0.28, 0.31, 0.33, 1.0]
        floor.condim = 3
        floor.friction = [1.0, 0.05, 0.05]

    if world.markers:
        # World is ENU, and a MuJoCo identity attitude is PX4 yaw +90 degrees
        # (plan section 3.6). Visible axes make that checkable in the viewer
        # rather than something to re-derive.
        for name, pos, rgba in (
            ("east_marker", [3.0, 0.0, 0.02], [1.0, 0.2, 0.2, 1.0]),
            ("north_marker", [0.0, 3.0, 0.02], [0.2, 1.0, 0.2, 1.0]),
        ):
            marker = spec.worldbody.add_site()
            marker.name = name
            marker.pos = pos
            marker.size = [0.05, 0.0, 0.0]
            marker.rgba = rgba


def _disable_discardvisual(urdf_text: str) -> str:
    """Inject ``<mujoco><compiler discardvisual="false"/></mujoco>`` into the URDF.

    MuJoCo's URDF parser defaults ``discardvisual`` to true and applies it *while
    parsing*: a link whose only geometry is ``<visual>`` loses it before the spec
    is handed back, so clearing the flag afterwards is too late -- the geometry is
    already gone.

    That matters because ``_name_geoms`` deliberately takes the CAD meshes out of
    collision (their convex hull wraps the whole vehicle, section 3.2). Left to
    the default, the arm links come back with no geometry at all, and the model
    still compiles, still has the right mass, and still flies -- there is nothing
    in a headless run that would reveal it.

    The ``<mujoco>`` element is a MuJoCo-specific URDF extension that other URDF
    consumers ignore, so the staged copy stays a valid URDF.
    """
    if "discardvisual" in urdf_text:
        return urdf_text
    extension = '  <mujoco><compiler discardvisual="false"/></mujoco>\n'
    marker = urdf_text.find("<robot")
    if marker < 0:
        raise ValueError("not a URDF: no <robot> element")
    insert = urdf_text.index(">", marker) + 1
    return urdf_text[:insert] + "\n" + extension + urdf_text[insert:]


def _inject_freejoint(spec: mujoco.MjSpec, base: Any) -> None:
    """Give the base its 6 DoF.

    MuJoCo's URDF parser welds the root link to the world (section 1), so
    without this the vehicle is immovable -- and a welded base does not error,
    it just never accelerates, which reads as a thrust problem.
    """
    if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in base.joints):
        return
    joint = base.add_joint()
    joint.name = "base_free"
    joint.type = mujoco.mjtJoint.mjJNT_FREE


def _name_geoms(spec: mujoco.MjSpec, sidecar: Sidecar) -> None:
    """Name the URDF's anonymous geoms, and take the CAD meshes out of collision.

    The URDF gives visual and collision the same mesh. Keeping that as collision
    geometry means the convex hull of 8 propellers wraps the vehicle, so takeoff,
    landing and arm motion all collide with the ground (section 3.2).
    """
    for body in spec.bodies:
        if body.name == "world":
            continue
        for index, geom in enumerate(body.geoms):
            if not geom.name:
                geom.name = f"{body.name}_visual{index}"
            if sidecar.collision_meshes_off and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.contype = 0
                geom.conaffinity = 0
                # Visual group, so the viewer's collision view stays readable.
                geom.group = 1
                # Mass comes from the URDF <inertial> block, which SolidWorks
                # filled from its own mass properties -- the whole reason to go
                # through CAD. Density must not add a second, conflicting mass.
                geom.density = 0.0


def _inject_sites(spec: mujoco.MjSpec, sidecar: Sidecar) -> None:
    """Add the rotor sites and the named sites (``imu``, optional ``ee``)."""
    for name, site in sidecar.sites.items():
        body = spec.body(site.body)
        if body is None:
            raise ValueError(f"sites.{name}: no body named {site.body!r}")
        added = body.add_site()
        added.name = name
        added.pos = list(site.pos)
        if site.quat is not None:
            added.quat = list(site.quat)
        elif name == IMU_SITE:
            # Hardcoded in section 3.3: the IMU site is aligned to the body
            # frame, and every frame conversion in frames.py assumes it.
            added.quat = [1.0, 0.0, 0.0, 0.0]

    base = spec.body(BASE_BODY)
    for index, rotor in enumerate(sidecar.rotors):
        # Deliberately on base_link and nowhere else: vehicle.py subtracts
        # base_link's body_ipos from site_pos, but site_pos is local to whichever
        # body owns the site. A rotor site on a child body raises no error, gets
        # the wrong lever arm, and presents as attitude cross-coupling. This is
        # section 2.2's hazard with no test coverage in src/, so the conversion
        # is where it gets prevented.
        site = base.add_site()
        site.name = f"{ROTOR_PREFIX}{index}"
        site.pos = list(rotor.pos)
        # Explicit zaxis costs nothing now and is what a tilt-rotor would need
        # later (section 5). vehicle.py hardcodes thrust along body +z today.
        site.alt.zaxis = list(rotor.zaxis)
        site.rgba = [0.9, 0.3, 0.3, 1.0] if rotor.spin > 0 else [0.3, 0.3, 0.9, 1.0]


def _inject_sensors(spec: mujoco.MjSpec) -> None:
    """Add ``imu_accel`` / ``imu_gyro``, which ``sim.py`` reads by name."""
    for name, kind in (
        ("imu_accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
        ("imu_gyro", mujoco.mjtSensor.mjSENS_GYRO),
    ):
        sensor = spec.add_sensor()
        sensor.name = name
        sensor.type = kind
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = IMU_SITE


def _inject_arm(spec: mujoco.MjSpec, sidecar: Sidecar) -> None:
    """Set joint limits and dynamics, and add one position actuator per joint."""
    found = sorted(
        (j.name for j in spec.joints if j.name.startswith(ARM_JOINT_PREFIX)),
        key=_joint_sort_key,
    )
    missing = set(found) - set(sidecar.joints)
    if missing:
        raise ValueError(
            f"the URDF has {sorted(missing)} but the sidecar does not. "
            f"SolidWorks exports lower=upper=0, which MuJoCo reads as "
            f"*unlimited* -- so an unconfigured joint is a free joint with a "
            f"meaningless ctrlrange, not a locked one"
        )
    unknown = set(sidecar.joints) - set(found)
    if unknown:
        raise ValueError(f"sidecar configures joints not in the URDF: {sorted(unknown)}")

    for index, name in enumerate(found):
        spec_joint = sidecar.joints[name]
        joint = spec.joint(name)
        joint.range = [spec_joint.lower, spec_joint.upper]
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.damping = [spec_joint.damping, 0.0, 0.0]
        joint.armature = spec_joint.armature
        joint.frictionloss = spec_joint.frictionloss

        actuator = spec.add_actuator()
        actuator.name = f"{ARM_ACTUATOR_PREFIX}{index}"
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        # An explicit PD, which is what <position kp= kv=> expands to. Written
        # out rather than via a default class so the gains are visible in the
        # generated XML next to the joint they drive.
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm = [spec_joint.kp] + [0.0] * 9
        actuator.biasprm = [0.0, -spec_joint.kp, -spec_joint.kv] + [0.0] * 7
        # ctrlrange mirrors the joint range: arm_cmd.values[i] is an absolute
        # angle in radians (section 3.4), so the command domain is the joint's.
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.ctrlrange = [spec_joint.lower, spec_joint.upper]
        if spec_joint.force_range is not None:
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = list(spec_joint.force_range)


def _joint_sort_key(name: str) -> tuple[int, str]:
    """Sort ``arm_joint2`` before ``arm_joint10``; index order is PX4-like."""
    tail = name[len(ARM_JOINT_PREFIX):]
    return (int(tail), "") if tail.isdigit() else (1 << 30, name)


def _inject_collisions(spec: mujoco.MjSpec, sidecar: Sidecar) -> None:
    """Add the hand-written collision primitives (section 3.2)."""
    types = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }
    for primitive in sidecar.collisions:
        body = spec.body(primitive.body)
        if body is None:
            raise ValueError(
                f"collisions[{primitive.name}]: no body named {primitive.body!r}"
            )
        if primitive.type not in types:
            raise ValueError(
                f"collisions[{primitive.name}]: type {primitive.type!r} not in "
                f"{sorted(types)}"
            )
        geom = body.add_geom()
        geom.name = primitive.name
        geom.type = types[primitive.type]
        if primitive.fromto is not None:
            geom.fromto = list(primitive.fromto)
        else:
            geom.pos = list(primitive.pos)
        if primitive.size:
            geom.size = list(primitive.size) + [0.0] * (3 - len(primitive.size))
        if primitive.quat is not None:
            geom.quat = list(primitive.quat)
        geom.contype = 1
        geom.conaffinity = 1
        geom.group = 3
        # Massless: the URDF <inertial> blocks already account for all of it.
        # A collision primitive that also carried mass would double-count.
        geom.density = 0.0
        geom.rgba = [0.4, 0.8, 0.4, 0.3]


def _apply_option(spec: mujoco.MjSpec, sidecar: Sidecar) -> None:
    """Apply ``<option>`` overrides.

    ``timestep`` is deliberately not settable: ``sim.py`` overwrites it from
    ``--physics-rate`` on every load, so honouring it here would be a lie.
    """
    for key, value in sidecar.option.items():
        if key == "timestep":
            raise ValueError(
                "option.timestep has no effect: sim.py sets model.opt.timestep "
                "from --physics-rate on every load (section 3.5)"
            )
        if key == "integrator":
            integrators = {
                "Euler": mujoco.mjtIntegrator.mjINT_EULER,
                "RK4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            }
            if value not in integrators:
                raise ValueError(
                    f"option.integrator {value!r} not in {sorted(integrators)}"
                )
            spec.option.integrator = integrators[value]
        elif key == "gravity":
            spec.option.gravity = list(value)
        elif hasattr(spec.option, key):
            setattr(spec.option, key, value)
        else:
            raise ValueError(f"option.{key} is not a MuJoCo option field")


# --- self-check -----------------------------------------------------------


@dataclass
class CheckResult:
    lines: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, text: str) -> None:
        self.lines.append(f"  ok    {text}")

    def fail(self, text: str) -> None:
        self.lines.append(f"  FAIL  {text}")
        self.failures.append(text)

    def warn(self, text: str) -> None:
        self.lines.append(f"  warn  {text}")
        self.warnings.append(text)


def self_check(model: mujoco.MjModel, sidecar: Sidecar) -> CheckResult:
    """Everything section 6 requires the script to verify and print.

    Frame orientation (FLU) is the one thing not checkable here: nothing in the
    model distinguishes a correct reference frame from one rotated 90 degrees.
    That rests on section 3.1 and a manual review.
    """
    result = CheckResult()
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    if base_id < 0:
        result.fail(f"no body named {BASE_BODY!r}")
        return result

    _check_freejoint(model, base_id, result)
    _check_rotor_sites(model, base_id, sidecar, result)
    _check_sensors(model, result)
    _check_actuators(model, sidecar, result)
    _check_mass(model, base_id, sidecar, result)
    _check_geometry(model, sidecar, result)
    _check_thrust(model, base_id, sidecar, result)
    return result


def _check_thrust(
    model: mujoco.MjModel, base_id: int, sidecar: Sidecar, result: CheckResult
) -> None:
    """Thrust/weight and the hover command the motor numbers imply.

    Both are things PX4 has to agree with, and neither is visible from the MJCF
    alone -- ``MPC_THR_HOVER`` is set by ``omega_idle / omega_max`` and
    thrust-to-weight and nothing else (section 2.5). Printing them here is what
    makes a mismatch against the airframe file a thing you notice now rather
    than in flight.
    """
    if any(rotor.c_t is None for rotor in sidecar.rotors):
        result.warn(
            "c_t not given for every rotor: vehicle.py auto-calibrates it from "
            "mass to thrust/weight 4.0 and warns on load (section 2.4). Thrust "
            "is then fabricated, not measured"
        )
        return

    mass = float(model.body_subtreemass[base_id])
    gravity = float(abs(model.opt.gravity[2]))
    weight = mass * gravity
    c_t = np.array([float(r.c_t) for r in sidecar.rotors])
    # omega_max is mandatory alongside c_t (RotorSpec enforces it).
    omega_max = np.array([float(r.omega_max) for r in sidecar.rotors])
    full = float(np.sum(c_t * omega_max ** 2))
    ratio = full / weight
    result.ok(
        f"thrust/weight {ratio:.3f} at full command "
        f"({full:.1f} N against {weight:.1f} N; "
        f"{full / len(c_t) / gravity:.3f} kgf per rotor)"
    )
    if ratio < 1.0:
        result.fail(
            f"thrust/weight is {ratio:.3f}: the vehicle cannot hover. Check the "
            f"mass against c_t and omega_max"
        )
        return
    if ratio < 1.8:
        result.warn(
            f"thrust/weight {ratio:.3f} leaves little control margin; 2.0 is a "
            f"common floor for a manipulator platform"
        )

    idle = sidecar.omega_idle
    if idle is None:
        result.warn(
            "omega_idle not set: vehicle.py's placeholder (100 rad/s) applies, "
            "and it moves the hover command, hence MPC_THR_HOVER"
        )
        return
    span = omega_max - idle
    a = float(np.sum(c_t * span ** 2))
    b = float(np.sum(2.0 * c_t * idle * span))
    c = float(np.sum(c_t * idle ** 2)) - weight
    hover = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    result.ok(
        f"hover command {hover:.4f} -> MPC_THR_HOVER {hover ** 2:.4f} at "
        f"THR_MDL_FAC 1 (idle thrust {100.0 * (c + weight) / weight:.1f}% of weight)"
    )
    if not 0.2 <= hover <= 0.8:
        result.warn(
            f"hover command {hover:.3f} is far from mid-stick; check omega_idle "
            f"and omega_max"
        )


def _check_geometry(
    model: mujoco.MjModel, sidecar: Sidecar, result: CheckResult
) -> None:
    """The CAD meshes survived, and something can actually touch the ground.

    Both failure modes here are silent. ``discardvisual`` defaults to true on the
    URDF path and deletes every non-colliding geom, so taking the meshes out of
    collision can delete them outright -- the model still compiles and flies,
    with no visual geometry at all. And a vehicle with no colliding geom falls
    through the floor rather than landing on it.
    """
    if model.nmesh == 0:
        result.fail(
            "no meshes in the compiled model: the CAD geometry was dropped "
            "(compiler.discardvisual deletes non-colliding geoms)"
        )
    else:
        result.ok(f"{model.nmesh} CAD mesh(es) retained as visual geometry")

    # Count only the vehicle's own geoms: the floor collides too, and including
    # it would mask a vehicle that has no collision geometry at all.
    vehicle = [
        index for index in range(model.ngeom)
        if int(model.body_weldid[model.geom_bodyid[index]]) != 0
        and int(model.geom_contype[index]) != 0
    ]
    if not vehicle:
        result.fail(
            "no colliding geom on the vehicle: it would fall through the floor. "
            "Add collision primitives to the sidecar (section 3.2)"
        )
    else:
        result.ok(f"{len(vehicle)} colliding geom(s) on the vehicle")
    if sidecar.collisions and len(vehicle) != len(sidecar.collisions):
        result.warn(
            f"{len(vehicle)} colliding vehicle geoms for "
            f"{len(sidecar.collisions)} configured primitives; a CAD mesh may "
            f"still be in collision"
        )

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if sidecar.world.floor and floor_id < 0:
        result.fail("world.floor is set but no 'floor' geom was emitted")
    elif not sidecar.world.floor:
        result.warn(
            "world.floor is false: nothing to take off from or land on, and the "
            "fall reads as a thrust deficit"
        )
    else:
        result.ok("ground plane present")


def _check_freejoint(model: mujoco.MjModel, base_id: int, result: CheckResult) -> None:
    count = int(model.body_jntnum[base_id])
    if count == 0:
        result.fail(f"{BASE_BODY} has no joint: MuJoCo welded it to the world")
        return
    first = int(model.body_jntadr[base_id])
    if model.jnt_type[first] != mujoco.mjtJoint.mjJNT_FREE:
        result.fail(f"{BASE_BODY}'s first joint is not a freejoint")
    else:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, first)
        result.ok(f"freejoint {name!r} is {BASE_BODY}'s first joint")


def _check_rotor_sites(
    model: mujoco.MjModel, base_id: int, sidecar: Sidecar, result: CheckResult
) -> None:
    """Contiguous, on ``base_link``, and as many as ``spin`` declares."""
    found = 0
    while mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{ROTOR_PREFIX}{found}"
    ) >= 0:
        found += 1
    expected = len(sidecar.rotors)
    if found != expected:
        # The scan in vehicle.py stops at the first missing index, so a gap
        # reads as a smaller vehicle with no error at all.
        result.fail(
            f"scanned {found} contiguous {ROTOR_PREFIX}N sites but the sidecar "
            f"declares {expected}; a gap reads as a smaller vehicle silently"
        )
    else:
        result.ok(f"{found} contiguous {ROTOR_PREFIX}N sites")

    offenders = []
    for index in range(found):
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{ROTOR_PREFIX}{index}"
        )
        if int(model.site_bodyid[site_id]) != base_id:
            owner = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site_id])
            )
            offenders.append(f"{ROTOR_PREFIX}{index} on {owner}")
    if offenders:
        result.fail(
            f"rotor sites not on {BASE_BODY}: {', '.join(offenders)}. "
            f"vehicle.py would use the wrong lever arm, silently (section 2.2)"
        )
    elif found:
        result.ok(f"all rotor sites are direct children of {BASE_BODY}")

    imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, IMU_SITE)
    if imu_id < 0:
        result.fail(f"no {IMU_SITE!r} site")
    else:
        quat = np.asarray(model.site_quat[imu_id], dtype=np.float64)
        if not np.allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-9):
            result.fail(
                f"{IMU_SITE} quat is {quat.round(6).tolist()}, not identity; "
                f"every conversion in frames.py assumes body alignment"
            )
        else:
            result.ok(f"{IMU_SITE} site is aligned to the body frame")


def _check_sensors(model: mujoco.MjModel, result: CheckResult) -> None:
    for name in ("imu_accel", "imu_gyro"):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            result.fail(f"no sensor named {name!r}; sim.py reads it every frame")
        elif int(model.sensor_dim[sensor_id]) != 3:
            result.fail(f"sensor {name} has dim {model.sensor_dim[sensor_id]}, not 3")
        else:
            result.ok(f"sensor {name} present, 3-axis")


def _check_actuators(
    model: mujoco.MjModel, sidecar: Sidecar, result: CheckResult
) -> None:
    """Arm actuators must be contiguous from 0: ``arm_cmd`` indexes them."""
    found = 0
    while mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ARM_ACTUATOR_PREFIX}{found}"
    ) >= 0:
        found += 1
    if found != len(sidecar.joints):
        result.fail(
            f"{found} contiguous {ARM_ACTUATOR_PREFIX}N actuators for "
            f"{len(sidecar.joints)} configured joints"
        )
    elif found:
        result.ok(
            f"{found} contiguous {ARM_ACTUATOR_PREFIX}N position actuators"
        )
    for index in range(found):
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ARM_ACTUATOR_PREFIX}{index}"
        )
        low, high = model.actuator_ctrlrange[actuator_id]
        if not high > low:
            result.fail(
                f"{ARM_ACTUATOR_PREFIX}{index} ctrlrange is [{low}, {high}]"
            )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_low, joint_high = model.jnt_range[joint_id]
        if not (math.isclose(low, joint_low, abs_tol=1e-9)
                and math.isclose(high, joint_high, abs_tol=1e-9)):
            result.warn(
                f"{ARM_ACTUATOR_PREFIX}{index} ctrlrange [{low:.4f}, {high:.4f}] "
                f"differs from its joint range [{joint_low:.4f}, {joint_high:.4f}]"
            )
        if int(model.jnt_limited[joint_id]) == 0:
            result.fail(
                f"joint of {ARM_ACTUATOR_PREFIX}{index} is unlimited; "
                f"SolidWorks' 0/0 export compiles to exactly this"
            )
        dof = int(model.jnt_dofadr[joint_id])
        if model.dof_armature[dof] <= 0.0:
            result.warn(
                f"{ARM_ACTUATOR_PREFIX}{index}'s joint has zero armature; a "
                f"stiff position actuator on an inertialess hinge oscillates"
            )
        if model.dof_damping[dof] <= 0.0:
            result.warn(f"{ARM_ACTUATOR_PREFIX}{index}'s joint has zero damping")


def _check_mass(
    model: mujoco.MjModel, base_id: int, sidecar: Sidecar, result: CheckResult
) -> None:
    """Total mass, CoM and inertia, to cross-check against SolidWorks."""
    total = float(model.body_subtreemass[base_id])
    result.ok(f"subtree mass {total:.4f} kg")
    if sidecar.expected_mass is not None:
        error = abs(total - float(sidecar.expected_mass))
        tolerance = float(sidecar.mass_tolerance)
        if error > tolerance:
            result.fail(
                f"subtree mass {total:.4f} kg differs from expected_mass "
                f"{sidecar.expected_mass:.4f} kg by {error:.4f} (tolerance "
                f"{tolerance}). A part with no material assigned is the usual "
                f"cause, and mass sets the auto-calibrated c_t"
            )
        else:
            result.ok(f"mass matches expected_mass to {error:.5f} kg")

    com = _subtree_com(model, base_id)
    result.ok(
        f"whole-vehicle CoM in {BASE_BODY} frame "
        f"({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}) m"
    )
    if sidecar.expected_com is not None:
        error = float(np.abs(com - np.asarray(sidecar.expected_com)).max())
        if error > 0.005:
            result.warn(
                f"CoM differs from expected_com by {error:.4f} m "
                f"(expected {tuple(round(v, 4) for v in sidecar.expected_com)})"
            )
        else:
            result.ok(f"CoM matches expected_com to {error:.5f} m")

    inertia = np.asarray(model.body_inertia[base_id], dtype=np.float64)
    result.ok(
        f"{BASE_BODY} principal inertia (about its own CoM) "
        f"{inertia.round(6).tolist()} kg m^2"
    )


def _subtree_com(model: mujoco.MjModel, base_id: int) -> NDArray[np.float64]:
    """Mass-weighted CoM of ``base_id``'s subtree, in ``base_id``'s frame.

    Computed at the home configuration via ``mj_forward``, then mapped back into
    the base frame -- so a moving arm does change it, which is the point of
    printing it next to the CAD figure.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    total = 0.0
    weighted = np.zeros(3)
    stack = [base_id]
    while stack:
        body = stack.pop()
        mass = float(model.body_mass[body])
        total += mass
        weighted += mass * np.asarray(data.xipos[body], dtype=np.float64)
        stack.extend(
            int(child) for child in range(model.nbody)
            if int(model.body_parentid[child]) == body and child != body
        )
    if total <= 0.0:
        return np.zeros(3)
    world = weighted / total
    origin = np.asarray(data.xpos[base_id], dtype=np.float64)
    rotation = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    return rotation.T @ (world - origin)


# --- reporting ------------------------------------------------------------


def rotor_table(sidecar: Sidecar) -> str:
    """The rotor block, as a table to compare against the PX4 airframe file.

    Also prints the ``spin`` tuple ready to paste into a ``RotorModel``: its
    length is what declares the expected rotor count, and a mismatch is a hard
    error in ``vehicle.py`` rather than a truncation.
    """
    lines = [
        "  idx  position (FLU)               deck    spin  km       c_t         w_max  label",
        "  ---  ---------------------------  ------  ----  -------  ----------  -----  -----",
    ]
    for index, rotor in enumerate(sidecar.rotors):
        x, y, z = rotor.pos
        c_t = f"{rotor.c_t:.4e}" if rotor.c_t is not None else "auto".rjust(10)
        omega_max = f"{rotor.omega_max:.0f}" if rotor.omega_max is not None else "  --"
        lines.append(
            f"  {index:<3d}  ({x:+.4f}, {y:+.4f}, {z:+.4f})  "
            f"{rotor.deck or '-':<6s}  {'CCW' if rotor.spin > 0 else 'CW':<4s}  "
            f"{rotor.km:<7.5f}  {c_t}  {omega_max:>5s}  {rotor.label}"
        )
    spin = ", ".join(f"{r.spin:+d}" for r in sidecar.rotors)
    lines.append("")
    lines.append(f"  RotorModel(spin=({spin}))")
    lines.append(
        "  PX4 CA_ROTOR*_PY is the negation of py above (FLU -> FRD); "
        "PZ and the site z produce no torque (section 4)."
    )
    kms = {round(r.km, 6) for r in sidecar.rotors}
    if len(kms) == 1 and abs(kms.pop() - 0.05) > 1e-6:
        km = sidecar.rotors[0].km
        lines.append(
            f"  CA_ROTOR*_KM must be set to {km:.5f}: PX4's default is 0.05, and "
            f"leaving it there makes the\n  allocator expect {0.05 / km:.2f}x the "
            f"yaw torque these props produce -- flyable, but yaw authority is\n"
            f"  overestimated, which presents as sluggish yaw rather than as an error."
        )
    return "\n".join(lines)


def _describe_coaxial(sidecar: Sidecar) -> list[str]:
    """Pair rotors sharing an (x, y) hub, and check each pair's spins oppose.

    A coaxial pair spinning the same way produces no differential yaw and halves
    the layout's yaw authority. It is easy to wire up wrong and invisible in
    hover, so it gets checked rather than assumed (section 4).
    """
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, rotor in enumerate(sidecar.rotors):
        key = (round(rotor.pos[0] * 1000), round(rotor.pos[1] * 1000))
        buckets.setdefault(key, []).append(index)
    notes: list[str] = []
    for (x, y), members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        spins = [sidecar.rotors[i].spin for i in members]
        pair = "+".join(str(i) for i in members)
        label = f"({x / 1000:+.3f}, {y / 1000:+.3f})"
        if len(set(spins)) == 1:
            notes.append(
                f"  warn  coaxial pair ({pair}) at {label} spins the same way: "
                f"no differential yaw from this pair"
            )
        else:
            notes.append(f"  ok    coaxial pair ({pair}) at {label}, spins oppose")
    return notes


def _print_meshes(reports: list[MeshReport]) -> None:
    print("\nmeshes (visual only; collision uses the sidecar's primitives)")
    for report in sorted(reports, key=lambda r: -r.source_faces):
        ratio = report.output_faces / max(report.source_faces, 1)
        note = "" if report.source_faces <= report.output_faces else (
            f"  {ratio:5.1%} of source, bbox error {report.bbox_error * 1000:.1f} mm"
        )
        print(
            f"  {report.name:<14s} {report.source_faces:>7d} -> "
            f"{report.output_faces:>6d} faces{note}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="urdf_to_mjcf.py",
        description=(
            "Convert a SolidWorks URDF export plus a sidecar YAML into the MJCF "
            "this simulator loads (MODELING_CONVENTIONS.md section 6)."
        ),
    )
    parser.add_argument("sidecar", type=Path, help="the .conversion.yaml")
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate and report without writing the MJCF or the meshes",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="override the sidecar's output path",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    sidecar = load_sidecar(args.sidecar.resolve())
    if args.output is not None:
        sidecar.output = args.output.resolve()
    if not sidecar.urdf.is_file():
        raise FileNotFoundError(f"urdf not found: {sidecar.urdf}")

    write = not args.check_only
    mesh_dir = sidecar.output.parent / sidecar.mesh_dir
    print(f"urdf    {sidecar.urdf}")
    print(f"output  {sidecar.output}{'  (not written: --check-only)' if not write else ''}")

    urdf_text = sidecar.urdf.read_text(encoding="utf-8")
    # Meshes are staged even under --check-only: MjSpec has to read them to
    # compile, and the raw export exceeds MuJoCo's decoder limit.
    mesh_dir.mkdir(parents=True, exist_ok=True)
    reports, mapping = prepare_meshes(sidecar, urdf_text, mesh_dir, write=True)
    spec = build_spec(sidecar, mesh_dir, mapping)
    model = spec.compile()

    print("\nrotors")
    print(rotor_table(sidecar))
    coaxial = _describe_coaxial(sidecar)
    if coaxial:
        print("\ncoaxial pairing")
        print("\n".join(coaxial))

    result = self_check(model, sidecar)
    print("\nself-check")
    print("\n".join(result.lines))
    _print_meshes(reports)

    print(
        "\nnot checkable here: the FLU frame convention (section 3.1). Nothing in "
        "the model distinguishes a correct reference frame from a rotated one --\n"
        "confirm in CAD that X is forward, Y is left, Z is up, and that the "
        "origin is at the rotor symmetry centre."
    )
    if sidecar.omega_idle is None:
        print(
            "\nomega_idle is not set, so vehicle.py's placeholder stands and "
            "warns on every load (section 2.4)."
        )
    if result.warnings:
        print(f"\n{len(result.warnings)} warning(s).")
    if result.failures:
        print(f"\n{len(result.failures)} check(s) FAILED; not writing the MJCF.")
        return 1

    if write:
        write_model(spec, sidecar, mesh_dir)
        print(f"\nwrote {sidecar.output}")
        print(f"wrote {len(reports)} mesh(es) to {mesh_dir}")
    return 0


def write_model(spec: mujoco.MjSpec, sidecar: Sidecar, mesh_dir: Path) -> None:
    """Emit the MJCF, then verify it loads standalone from its own directory.

    ``meshdir`` is set only now: during conversion the staged URDF lives in the
    mesh directory so bare filenames resolve, but the emitted XML sits one level
    up and needs the ``<compiler meshdir>`` hop. Reloading from disk is what
    proves the written file works for ``sim.py``, rather than only the in-memory
    spec working here.
    """
    sidecar.output.parent.mkdir(parents=True, exist_ok=True)
    # to_xml() re-resolves every mesh file, and it resolves them against the
    # spec's own modelfiledir -- which is the staging directory, not the output's.
    # An absolute meshdir is what makes both the serialization here and the
    # reload below find the same files.
    spec.meshdir = str(mesh_dir)
    xml = spec.to_xml()
    # Emit the portable relative form, so the model directory can be moved or
    # copied without carrying this machine's paths.
    xml = xml.replace(f'meshdir="{mesh_dir}"', f'meshdir="{sidecar.mesh_dir}"')
    xml = xml.replace(f'meshdir="{mesh_dir}/"', f'meshdir="{sidecar.mesh_dir}"')
    sidecar.output.write_text(_header(sidecar) + xml, encoding="utf-8")

    # Reload from disk: that is what proves the written file works for sim.py,
    # rather than only the in-memory spec working here.
    reloaded = mujoco.MjModel.from_xml_path(str(sidecar.output))
    if reloaded.nmesh == 0 and spec.meshes:
        raise ValueError(
            f"{sidecar.output}: reloaded without its meshes; check that "
            f"meshdir={sidecar.mesh_dir!r} is right relative to the output"
        )


def _header(sidecar: Sidecar) -> str:
    """A provenance banner, since the file is generated and must not be edited."""
    spin = ", ".join(f"{r.spin:+d}" for r in sidecar.rotors)
    notes = f"\n\n  {sidecar.notes.strip()}" if sidecar.notes.strip() else ""
    return (
        "<!--\n"
        "  GENERATED by scripts/urdf_to_mjcf.py. Do not edit by hand: the next\n"
        "  CAD export overwrites it. Change the sidecar YAML instead.\n\n"
        f"  urdf     {sidecar.urdf.name}\n"
        f"  rotors   {len(sidecar.rotors)}, spin=({spin})\n"
        f"  joints   {len(sidecar.joints)}\n\n"
        "  Frames (IMPLEMENTATION_PLAN.md 3.6, binding on every model):\n"
        "    body is FLU  -- +x forward, +y left, +z up\n"
        "    world is ENU -- +x east, +y north, +z up\n"
        "  timestep here is overridden by --physics-rate on load."
        f"{notes}\n"
        "-->\n"
    )


if __name__ == "__main__":
    sys.exit(main())
