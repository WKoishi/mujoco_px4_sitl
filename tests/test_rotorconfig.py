"""Tests for the sidecar -> RotorModel path (src/mujoco_px4_sitl/rotorconfig.py).

No private geometry needed: every test writes a small sidecar to a tmp dir. The
focus is the errors that would otherwise be **silent at runtime** -- a dropped
coefficient leaving a placeholder in force, a partial c_t set mixing measured
rotors with mass-calibrated ones, and a rotors block that reached RotorModel
with the wrong length.

The X8's real numbers are not here. They live in the private sidecar, and
asserting them would put hub geometry in the public repo (AGENTS.md section 4).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mujoco_px4_sitl.rotorconfig import (
    RotorSpec,
    load_rotors,
    parse_rotor_entry,
    rotors_from_specs,
)

# Coefficients shaped like a real fit but not any real motor: the point is that
# c_t and omega_max travel together, not their values.
_CT = 1.0e-04
_WMAX = 600.0


def _rotor(spin, **kw):
    entry = {"pos": [0.3, -0.3, 0.0], "spin": spin, "c_t": _CT, "omega_max": _WMAX}
    entry.update(kw)
    return entry


def _write(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "x.conversion.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _x8(**top):
    spins = [1, -1, 1, -1, 1, -1, 1, -1]
    body = {"rotors": [_rotor(s) for s in spins]}
    body.update(top)
    return body


def test_loads_an_8_rotor_model(tmp_path):
    rotors = load_rotors(_write(tmp_path, _x8(omega_idle=57.6)))
    assert rotors.spin == (1, -1, 1, -1, 1, -1, 1, -1)
    assert rotors.c_t == (_CT,) * 8
    assert rotors.omega_max == (_WMAX,) * 8
    assert rotors.omega_idle == 57.6


def test_spin_accepts_cw_ccw_strings(tmp_path):
    """The sidecar is hand-written, and CW/CCW is what the wiring is labelled."""
    body = {"rotors": [_rotor("CCW"), _rotor("CW")]}
    assert load_rotors(_write(tmp_path, body)).spin == (1, -1)


def test_omega_idle_absent_leaves_the_placeholder(tmp_path):
    """Not an error: omega_idle is an ESC setting and may be unmeasured.

    vehicle.py's placeholder then applies, and it warns -- which is the signal
    the value was never supplied.
    """
    rotors = load_rotors(_write(tmp_path, _x8()))
    # The default is vehicle.py's, not one we invented here.
    from mujoco_px4_sitl.vehicle import OMEGA_IDLE_PLACEHOLDER

    assert rotors.omega_idle == OMEGA_IDLE_PLACEHOLDER


def test_c_t_without_omega_max_is_rejected(tmp_path):
    """The trap from MODELING_CONVENTIONS 2.5: only the product means anything.

    A measured c_t against the placeholder omega_max loads fine and rescales
    every thrust in the model.
    """
    body = {"rotors": [_rotor(1, omega_max=None)]}
    body["rotors"][0].pop("omega_max")
    with pytest.raises(ValueError, match="c_t given without omega_max"):
        load_rotors(_write(tmp_path, body))


def test_partial_c_t_is_rejected(tmp_path):
    """Mixing measured rotors with mass-calibrated ones is not a vehicle.

    The fallback divides the whole weight across every rotor, so a subset
    cannot be calibrated coherently.
    """
    rotors = [_rotor(1), _rotor(-1)]
    rotors[1].pop("c_t")
    rotors[1].pop("omega_max")
    with pytest.raises(ValueError, match="all or nothing"):
        load_rotors(_write(tmp_path, {"rotors": rotors}))


def test_misspelled_coefficient_is_rejected(tmp_path):
    """A silently dropped key leaves the placeholder in force."""
    body = {"rotors": [_rotor(1, omeg_max=600.0)]}
    with pytest.raises(ValueError, match="unknown key"):
        load_rotors(_write(tmp_path, body))


def test_bad_spin_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="spin must be"):
        load_rotors(_write(tmp_path, {"rotors": [_rotor(0)]}))
    with pytest.raises(ValueError, match="CW or CCW"):
        load_rotors(_write(tmp_path, {"rotors": [_rotor("widdershins")]}))


def test_non_sidecar_file_is_named_as_such(tmp_path):
    """Pointing --rotors at an MJCF is an easy mistake; say so plainly."""
    path = tmp_path / "model.xml"
    path.write_text("mujoco: not a sidecar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conversion sidecar"):
        load_rotors(path)


def test_runtime_path_ignores_urdf_and_meshes(tmp_path):
    """Flying must not depend on the CAD export being present.

    The sidecar names a URDF and a mesh dir that do not exist here. The
    conversion script would refuse; the runtime loader must not care.
    """
    body = _x8(urdf="../nowhere/robot.urdf", output="./out.xml", mesh_dir="meshes")
    assert len(load_rotors(_write(tmp_path, body)).spin) == 8


def test_omega_min_is_all_or_nothing():
    specs = [RotorSpec(pos=(0, 0, 0), spin=1, c_t=_CT, omega_max=_WMAX, omega_min=60.0),
             RotorSpec(pos=(0, 0, 0), spin=-1, c_t=_CT, omega_max=_WMAX)]
    with pytest.raises(ValueError, match="omega_min"):
        rotors_from_specs(specs)


def test_coefficients_are_per_rotor_tuples():
    """The coaxial discount (section 5) must be a sidecar edit, not a code change.

    So even uniform coefficients arrive as per-rotor tuples.
    """
    specs = [RotorSpec(pos=(0, 0, 0), spin=s, c_t=_CT, omega_max=_WMAX)
             for s in (1, -1, 1, -1)]
    specs[2].c_t = 0.8 * _CT
    specs[3].c_t = 0.8 * _CT
    rotors = rotors_from_specs(specs)
    assert rotors.c_t == (_CT, _CT, 0.8 * _CT, 0.8 * _CT)


def test_empty_rotors_block_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no 'rotors' block"):
        load_rotors(_write(tmp_path, {"rotors": []}))


def test_parse_rotor_entry_labels_by_index():
    spec = parse_rotor_entry({"pos": [0, 0, 0], "spin": 1}, 3, "w", "rotor")
    assert spec.label == "rotor3"


def test_a_rotor_radius_is_accepted_and_does_not_touch_thrust(tmp_path):
    """radius is geometry: it reaches the simulator through the MJCF's disc
    sites, so the runtime rotor block parses it and otherwise ignores it."""
    plain = load_rotors(_write(tmp_path, {"rotors": [_rotor("CCW"), _rotor("CW")]}))
    with_radius = load_rotors(_write(tmp_path, {"rotors": [
        _rotor("CCW", radius=0.2), _rotor("CW", radius=0.2),
    ]}))
    assert with_radius == plain
    assert parse_rotor_entry(_rotor("CCW", radius=0.2), 0, "x", "rotor").radius == 0.2
