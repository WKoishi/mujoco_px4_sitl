"""Configuration for the simulator process.

One dataclass, populated from defaults, then environment variables, then CLI
flags (later wins). No hardcoded paths outside :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _REPO_ROOT / "models" / "quad_x.xml"

# PX4's traditional SITL home (Zurich). PX4 does not read PX4_HOME_* on the
# mavlinksim path -- the origin is whatever our first HIL_STATE_QUATERNION
# carries (plan section 3.6) -- so this value alone defines the local frame.
DEFAULT_HOME_LAT = 47.397742
DEFAULT_HOME_LON = 8.545594
DEFAULT_HOME_ALT = 488.0


@dataclass
class Config:
    """Everything the simulator needs to run, in SI units."""

    # --- PX4 HIL transport -------------------------------------------------
    instance: int = 0
    hil_bind_host: str = "127.0.0.1"
    hil_port_base: int = 4560

    # --- Rates -------------------------------------------------------------
    # Must match IMU_INTEG_RATE, which px4-rc.simulator pins to 250 for every
    # posix simulator (plan section 5).
    imu_rate_hz: float = 250.0
    # Physics rate must be an integer multiple of the IMU rate (plan phase 3).
    physics_rate_hz: float = 1000.0
    speed_factor: float = 1.0

    # --- Lockstep loop tunables (plan section 3.2) -------------------------
    # Measured against a live PX4, not guessed. A brake timeout is pure wasted
    # wall clock under lockstep -- while we block, PX4's clock is frozen, so no
    # new actuator message can be produced -- hence a short timeout and a lead
    # large enough to make braking rare. See the table in the plan's 3.2.
    max_lead_frames: int = 32
    brake_timeout_s: float = 0.05
    status_interval_s: float = 5.0

    # --- Model -------------------------------------------------------------
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL)
    stub_physics: bool = False
    # Conversion sidecar supplying the rotor parameters. No default: a model
    # whose rotor count differs from RotorModel's quad default is rejected at
    # load, so forgetting this is loud rather than a silent fall back to
    # placeholder motors. The filled sidecar for the X8 lives in the private
    # repo alongside its MJCF (AGENTS.md section 4).
    rotors_path: Path | None = None

    # --- Phase 3 open-loop bring-up ----------------------------------------
    # Pin the vehicle so ground truth moves only in ways we dictate.
    hold_pose: bool = False
    hold_height: float = 2.0
    # MuJoCo-frame attitude to inject, as 321 Euler degrees about body axes.
    inject_attitude: tuple[float, float, float] | None = None

    # --- Geodetic origin ---------------------------------------------------
    home_lat: float = DEFAULT_HOME_LAT
    home_lon: float = DEFAULT_HOME_LON
    home_alt: float = DEFAULT_HOME_ALT

    # --- Side channel ------------------------------------------------------
    sidechannel_enabled: bool = True
    sidechannel_bind_host: str = "127.0.0.1"
    sidechannel_port_base: int = 14650
    sidechannel_rate_hz: float = 50.0

    # --- Misc --------------------------------------------------------------
    viewer: bool = False
    max_sim_time: float | None = None
    log_level: str = "INFO"

    @property
    def hil_port(self) -> int:
        """PX4 connects to 4560 + instance (px4-rc.mavlinksim)."""
        return self.hil_port_base + self.instance

    @property
    def sidechannel_port(self) -> int:
        return self.sidechannel_port_base + self.instance

    @property
    def imu_dt(self) -> float:
        return 1.0 / self.imu_rate_hz

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_rate_hz

    @property
    def steps_per_imu_frame(self) -> int:
        ratio = self.physics_rate_hz / self.imu_rate_hz
        n = int(round(ratio))
        if n < 1 or abs(ratio - n) > 1e-9:
            raise ValueError(
                f"physics_rate_hz ({self.physics_rate_hz}) must be an integer "
                f"multiple of imu_rate_hz ({self.imu_rate_hz})"
            )
        return n

    def validate(self) -> None:
        self.steps_per_imu_frame  # raises on a bad rate ratio
        if self.speed_factor <= 0.0:
            raise ValueError("speed_factor must be > 0")
        if self.max_lead_frames < 1:
            raise ValueError("max_lead_frames must be >= 1")
        if self.brake_timeout_s <= 0.0:
            raise ValueError("brake_timeout_s must be > 0 (wall clock)")
        if self.inject_attitude is not None and not self.hold_pose:
            # Without the pin, physics integrates the injected attitude away
            # immediately, so the flag would silently do nothing (plan phase 3
            # uses the two together).
            raise ValueError(
                "--inject-attitude requires --hold-pose: an unpinned vehicle "
                "integrates the injected attitude away on the first step"
            )
        if not self.stub_physics and not Path(self.model_path).is_file():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        if self.rotors_path is not None and not Path(self.rotors_path).is_file():
            raise FileNotFoundError(f"rotors sidecar not found: {self.rotors_path}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else Path(raw)


def _euler_triple(raw: str) -> tuple[float, float, float]:
    parts = [p for p in raw.replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected ROLL,PITCH,YAW in degrees")
    roll, pitch, yaw = (float(p) for p in parts)
    return roll, pitch, yaw


def build_parser() -> argparse.ArgumentParser:
    """CLI parser. Defaults come from the environment where one applies."""
    p = argparse.ArgumentParser(
        prog="python -m mujoco_px4_sitl",
        description="MuJoCo physics backend for PX4 SITL (MAVLink HIL, lockstep).",
    )
    p.add_argument(
        "-i", "--instance", type=int, default=_env_int("PX4_INSTANCE", 0),
        help="PX4 instance; HIL port is 4560+instance (default: %(default)s)",
    )
    p.add_argument("--hil-bind-host", default=os.environ.get("MUJOCO_SITL_BIND", "127.0.0.1"))
    p.add_argument(
        "--hil-port-base", type=int, default=_env_int("MUJOCO_SITL_HIL_PORT", 4560),
        help="HIL port is this plus --instance (default: %(default)s)",
    )
    p.add_argument(
        "-m", "--model", dest="model_path", type=Path, default=DEFAULT_MODEL,
        help="MuJoCo MJCF model (default: %(default)s)",
    )
    p.add_argument(
        "--rotors", dest="rotors_path", type=Path, default=_env_path("MUJOCO_SITL_ROTORS"),
        help=(
            "conversion sidecar supplying rotor parameters. Required for any "
            "model whose rotor count is not 4 (default: $MUJOCO_SITL_ROTORS, "
            "else vehicle.py's quad placeholders)"
        ),
    )
    p.add_argument("--imu-rate", dest="imu_rate_hz", type=float, default=250.0,
                   help="must match IMU_INTEG_RATE (default: %(default)s)")
    p.add_argument("--physics-rate", dest="physics_rate_hz", type=float, default=1000.0,
                   help="integer multiple of --imu-rate (default: %(default)s)")
    p.add_argument("-s", "--speed-factor", type=float, default=1.0,
                   help="simulated seconds per wall second (default: %(default)s)")
    p.add_argument("--max-lead-frames", type=int, default=32,
                   help="IMU frames outstanding before braking (default: %(default)s)")
    p.add_argument("--brake-timeout", dest="brake_timeout_s", type=float, default=0.05,
                   help="wall-clock brake timeout in seconds (default: %(default)s)")
    p.add_argument("--status-interval", dest="status_interval_s", type=float, default=5.0,
                   help="seconds between sim/wall ratio log lines (default: %(default)s)")
    p.add_argument("--stub-physics", action="store_true",
                   help="phase 1: hardcoded level-hover state, no mj_step")
    p.add_argument("--hold-pose", action="store_true",
                   help="phase 3: pin the vehicle in place, ignore actuators")
    p.add_argument("--hold-height", type=float, default=2.0,
                   help="height to hold with --hold-pose (default: %(default)s)")
    p.add_argument(
        "--inject-attitude", type=_euler_triple, default=None,
        metavar="ROLL,PITCH,YAW",
        help="phase 3: hold this MuJoCo-frame attitude, 321 Euler degrees",
    )
    p.add_argument("--home-lat", type=float, default=_env_float("PX4_HOME_LAT", DEFAULT_HOME_LAT))
    p.add_argument("--home-lon", type=float, default=_env_float("PX4_HOME_LON", DEFAULT_HOME_LON))
    p.add_argument("--home-alt", type=float, default=_env_float("PX4_HOME_ALT", DEFAULT_HOME_ALT))
    p.add_argument("--no-sidechannel", dest="sidechannel_enabled", action="store_false")
    p.add_argument(
        "--sidechannel-port-base", type=int,
        default=_env_int("MUJOCO_SITL_SIDECHANNEL_PORT", 14650),
        help="side-channel port is this plus --instance (default: %(default)s)",
    )
    p.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (needs GL)")
    p.add_argument("--max-sim-time", type=float, default=None,
                   help="stop after N simulated seconds (testing)")
    p.add_argument("--log-level", default=os.environ.get("MUJOCO_SITL_LOG", "INFO"))
    return p


def config_from_args(argv: list[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    known = {f for f in Config.__dataclass_fields__}
    cfg = Config(**{k: v for k, v in vars(args).items() if k in known})
    cfg.validate()
    return cfg
