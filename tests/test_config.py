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
])
def test_rejected_configs_exit_cleanly_rather_than_traceback(argv, capsys):
    """User error should read like argparse's own, not like a crash. Exit 2 is
    what argparse uses for a bad argument, so match it.
    """
    from mujoco_px4_sitl.main import main

    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err
