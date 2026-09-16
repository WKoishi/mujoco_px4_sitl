"""CLI entry point.

Launchable as ``python -m mujoco_px4_sitl``, as the ``mujoco-px4-sitl`` script,
or from a ROS 2 launch file via ``ExecuteProcess`` -- no ROS 2 code here (plan
section 1).
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from .config import Config, config_from_args
from .loop import LockstepLoop
from .sidechannel import SideChannel
from .sim import MujocoPhysics, build_physics
from .transport import HilServer

_log = logging.getLogger("mujoco_px4_sitl")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(cfg: Config) -> int:
    _log.info(
        "mujoco_px4_sitl instance %d: HIL tcp://%s:%d%s",
        cfg.instance, cfg.hil_bind_host, cfg.hil_port,
        " (stub physics)" if cfg.stub_physics else f", model {cfg.model_path}",
    )

    physics = build_physics(cfg)
    # Bind before PX4 needs us: it retries connect() every 500 us, so either
    # start order works (plan 3.1).
    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    sidechannel = (
        SideChannel(cfg.sidechannel_bind_host, cfg.sidechannel_port)
        if cfg.sidechannel_enabled else None
    )

    view = None
    frame_hook = None
    if cfg.viewer:
        if isinstance(physics, MujocoPhysics):
            from .viewer import Viewer

            view = Viewer(physics.model, physics.data)
            frame_hook = view.sync
        else:
            _log.warning("--viewer has nothing to show with --stub-physics")

    loop = LockstepLoop(cfg, physics, server, sidechannel, frame_hook)

    def _shutdown(signum: int, _frame: FrameType | None) -> None:
        _log.info("signal %d received, shutting down", signum)
        loop.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run()
    except KeyboardInterrupt:
        _log.info("interrupted")
    finally:
        if view is not None:
            view.close()
        if sidechannel is not None:
            sidechannel.close()
        server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(argv)
    _configure_logging(cfg.log_level)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
