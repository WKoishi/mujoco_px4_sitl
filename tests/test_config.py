"""Config validation tests. No PX4 build, no MuJoCo model load."""

from __future__ import annotations

import pytest

from mujoco_px4_sitl.config import Config, config_from_args


def test_inject_attitude_without_hold_pose_is_rejected():
    """It would otherwise be a silent no-op: nothing applies the injected
    attitude unless the pose is pinned, and an unpinned vehicle integrates it
    away on the first step. Failing loudly beats a phase-3 attitude check that
    quietly measures identity.
    """
    cfg = Config(stub_physics=True, inject_attitude=(30.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="requires --hold-pose"):
        cfg.validate()


def test_inject_attitude_with_hold_pose_is_accepted():
    cfg = Config(stub_physics=True, hold_pose=True, inject_attitude=(30.0, 0.0, 0.0))
    cfg.validate()  # must not raise


def test_hold_pose_alone_is_accepted():
    """Pinning level, with no injected attitude, is a valid phase-3 rig."""
    Config(stub_physics=True, hold_pose=True).validate()


def test_the_cli_rejects_the_same_combination():
    """The check must sit in validate(), which the CLI path runs, rather than in
    the parser -- config files and direct construction go through it too.
    """
    with pytest.raises(ValueError, match="requires --hold-pose"):
        config_from_args(["--stub-physics", "--inject-attitude", "30,0,0"])


def test_ports_derive_from_the_instance():
    cfg = config_from_args(["--stub-physics", "--instance", "3"])
    assert cfg.hil_port == 4563  # PX4's own convention: 4560 + instance
    assert cfg.sidechannel_port == 14653


def test_port_bases_are_overridable_on_the_cli():
    cfg = config_from_args([
        "--stub-physics", "--instance", "2",
        "--hil-port-base", "5000", "--sidechannel-port-base", "15000",
    ])
    assert cfg.hil_port == 5002
    assert cfg.sidechannel_port == 15002


def test_port_bases_are_overridable_from_the_environment(monkeypatch):
    monkeypatch.setenv("MUJOCO_SITL_HIL_PORT", "6000")
    monkeypatch.setenv("MUJOCO_SITL_SIDECHANNEL_PORT", "16000")
    cfg = config_from_args(["--stub-physics"])
    assert cfg.hil_port == 6000
    assert cfg.sidechannel_port == 16000


def test_a_non_integer_rate_ratio_is_rejected():
    """The IMU frame must be a whole number of physics steps (plan phase 3)."""
    cfg = Config(stub_physics=True, imu_rate_hz=250.0, physics_rate_hz=1100.0)
    with pytest.raises(ValueError, match="integer multiple"):
        cfg.validate()


@pytest.mark.parametrize("argv", [
    ["--stub-physics", "--inject-attitude", "30,0,0"],
    ["--model", "/nonexistent/model.xml"],
    ["--stub-physics", "--physics-rate", "1100"],
    ["--stub-physics", "--rotors", "/nonexistent/x.conversion.yaml"],
])
def test_rejected_configs_exit_cleanly_rather_than_traceback(argv, capsys):
    """User error should read like argparse's own, not like a crash. Exit 2 is
    what argparse uses for a bad argument, so match it.
    """
    from mujoco_px4_sitl.main import main

    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err


def test_rotors_path_defaults_to_the_environment(monkeypatch):
    """run_sitl.sh passes the model by env var; rotors follows that pattern so
    the shell script needs no change to fly a non-quad.
    """
    monkeypatch.setenv("MUJOCO_SITL_ROTORS", "/tmp/some.conversion.yaml")
    from mujoco_px4_sitl.config import build_parser

    args = build_parser().parse_args([])
    assert str(args.rotors_path) == "/tmp/some.conversion.yaml"


def test_rotors_path_has_no_default():
    """A non-quad model must not silently fall back to placeholder motors.

    There is deliberately no default sidecar: the filled one is private
    (AGENTS.md section 4), and an 8-rotor model met by RotorModel's 4-entry spin
    raises rather than flying with the quad's numbers.
    """
    from mujoco_px4_sitl.config import build_parser

    assert build_parser().parse_args([]).rotors_path is None


def test_passing_both_a_rotor_model_and_a_sidecar_is_rejected(tmp_path):
    """One would be parsed and then discarded, so it is a contradiction rather
    than a precedence question.
    """
    from mujoco_px4_sitl.main import run
    from mujoco_px4_sitl.vehicle import RotorModel

    sidecar = tmp_path / "x.conversion.yaml"
    sidecar.write_text("rotors: [{pos: [0,0,0], spin: 1}]\n", encoding="utf-8")
    cfg = Config(stub_physics=True, rotors_path=sidecar)
    with pytest.raises(ValueError, match="both a RotorModel and --rotors"):
        run(cfg, RotorModel(spin=(1, -1, 1, -1)))


def test_arm_watchdog_flags_reach_the_config():
    cfg = config_from_args(
        ["--stub-physics", "--arm-timeout", "0.2", "--arm-on-timeout", "limp"]
    )
    assert cfg.arm_timeout_s == 0.2
    assert cfg.arm_on_timeout == "limp"


def test_the_arm_watchdog_is_on_by_default():
    """A crashed controller must not leave the arm on its last command forever
    unless someone asked for exactly that with --arm-timeout 0."""
    cfg = config_from_args(["--stub-physics"])
    assert cfg.arm_timeout_s > 0.0
    assert cfg.arm_on_timeout == "freeze"


def test_an_unknown_timeout_action_is_rejected_by_the_cli():
    with pytest.raises(SystemExit):
        config_from_args(["--stub-physics", "--arm-on-timeout", "hold"])
