"""Sidecar rotor block -> :class:`RotorModel`.

The conversion sidecar (MODELING_CONVENTIONS.md section 6) is the one place a
platform's measured motor numbers are written down. Two consumers read it:
``scripts/urdf_to_mjcf.py`` at conversion time, and this module at startup, via
``--rotors`` / ``MUJOCO_SITL_ROTORS``. **They share one parser**, which is why
:class:`RotorSpec` lives here rather than in the script -- a second parser for
the same file is how the two drift apart, and the drift is silent: a model whose
rotor count matches still flies, with the wrong thrust.

:func:`load_rotors` reads **only** the runtime fields -- ``rotors[]`` and
``omega_idle``. It deliberately does not touch ``urdf``, ``output`` or
``mesh_dir``, so flying does not depend on the CAD export being present. The
STLs are 7 MB of private geometry; the eight numbers needed to fly are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vehicle import RotorModel

# Sidecar keys this module understands. The full schema is larger -- the script
# owns the rest -- so an unknown key here is not an error, unlike inside the
# rotors block where it means a dropped coefficient.
_ROTOR_KEYS = {
    "pos", "spin", "c_t", "km", "omega_max", "omega_min", "zaxis", "deck", "label",
}


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


def parse_rotor_entry(entry: Any, index: int, where: str, prefix: str) -> RotorSpec:
    """One ``rotors[]`` entry -> :class:`RotorSpec`. Raises on anything odd."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(entry).__name__}")
    unknown = sorted(set(entry) - _ROTOR_KEYS)
    if unknown:
        # A misspelled coefficient that is silently dropped leaves the
        # placeholder in force, which is exactly the failure this schema exists
        # to prevent.
        raise ValueError(
            f"{where}: unknown key(s) {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(_ROTOR_KEYS))}"
        )
    if "pos" not in entry:
        raise ValueError(f"{where}: missing required key 'pos'")
    if "spin" not in entry:
        raise ValueError(f"{where}: missing required key 'spin'")

    spin_raw = entry["spin"]
    if isinstance(spin_raw, str):
        text = spin_raw.strip().upper()
        if text not in ("CW", "CCW"):
            raise ValueError(f"{where}: spin string must be CW or CCW")
        spin = 1 if text == "CCW" else -1
    else:
        spin = int(spin_raw)

    return RotorSpec(
        pos=tuple(float(v) for v in entry["pos"]),
        spin=spin,
        c_t=entry.get("c_t"),
        km=float(entry.get("km", 0.05)),
        omega_max=entry.get("omega_max"),
        omega_min=entry.get("omega_min"),
        zaxis=tuple(float(v) for v in entry.get("zaxis", (0.0, 0.0, 1.0))),
        deck=str(entry.get("deck", "")),
        label=str(entry.get("label", f"{prefix}{index}")),
    )


def rotors_from_specs(
    specs: list[RotorSpec], omega_idle: float | None = None
) -> RotorModel:
    """Build the runtime :class:`RotorModel` from parsed specs.

    Every coefficient is passed as a per-rotor tuple even when uniform, so the
    coaxial lower-deck ``c_t`` discount (MODELING_CONVENTIONS.md section 5) is a
    sidecar edit and not a code change.

    ``c_t`` is all-or-nothing: a partial set would mean mixing measured rotors
    with mass-calibrated ones on one vehicle, and the calibration divides the
    whole weight across all rotors, so the mixture is not meaningful.
    """
    if not specs:
        raise ValueError("no rotors in sidecar")

    measured = [s.c_t is not None for s in specs]
    if any(measured) and not all(measured):
        missing = [i for i, m in enumerate(measured) if not m]
        raise ValueError(
            f"c_t given for some rotors but not {missing}. It is all or "
            f"nothing: the fallback calibrates c_t from the model's mass "
            f"across every rotor, so a mixture is not a meaningful vehicle"
        )

    kwargs: dict[str, Any] = {"spin": tuple(int(s.spin) for s in specs)}
    if all(measured):
        kwargs["c_t"] = tuple(float(s.c_t) for s in specs)  # type: ignore[arg-type]
        # RotorSpec guarantees omega_max alongside c_t.
        kwargs["omega_max"] = tuple(float(s.omega_max) for s in specs)  # type: ignore[arg-type]
    kwargs["km"] = tuple(float(s.km) for s in specs)
    if any(s.omega_min is not None for s in specs):
        if any(s.omega_min is None for s in specs):
            raise ValueError("omega_min must be given for every rotor or none")
        kwargs["omega_min"] = tuple(float(s.omega_min) for s in specs)  # type: ignore[arg-type]
    if omega_idle is not None:
        kwargs["omega_idle"] = float(omega_idle)
    return RotorModel(**kwargs)


def load_rotors(path: Path | str) -> RotorModel:
    """Read a conversion sidecar's rotor block into a :class:`RotorModel`.

    Reads ``rotors[]`` and ``omega_idle`` and nothing else. Importing yaml
    lazily keeps it off the import path of a simulator run that does not use
    ``--rotors``, so the dependency stays where the sidecar is.
    """
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    entries = raw.get("rotors")
    if not entries:
        raise ValueError(
            f"{path}: no 'rotors' block. This must be a conversion sidecar "
            f"(MODELING_CONVENTIONS.md section 6), not an MJCF or a URDF"
        )
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'rotors' must be a list")

    specs = [
        parse_rotor_entry(entry, i, f"{path}: rotors[{i}]", "rotor")
        for i, entry in enumerate(entries)
    ]
    idle = raw.get("omega_idle")
    return rotors_from_specs(specs, None if idle is None else float(idle))
