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
from .rotorconfig import load_rotors
from .sidechannel import SideChannel
from .sim import MujocoPhysics, build_physics
from .transport import HilServer
from .vehicle import RotorModel

_log = logging.getLogger("mujoco_px4_sitl")
_PROG = "mujoco_px4_sitl"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(cfg: Config, rotors: RotorModel | None = None) -> int:
    """Run the simulator. ``rotors`` overrides ``cfg.rotors_path``.

    Two ways in, because there are two callers. The CLI gives a sidecar path and
    this function parses it. An in-process driver -- a research script that wants
    ``MjData`` rather than the side channel -- passes a built ``RotorModel`` and
    skips the file entirely. Passing both is a contradiction rather than a
    precedence question, so it raises.
    """
    _log.info(
        "mujoco_px4_sitl instance %d: HIL tcp://%s:%d%s",
        cfg.instance, cfg.hil_bind_host, cfg.hil_port,
        " (stub physics)" if cfg.stub_physics else f", model {cfg.model_path}",
    )
    if rotors is not None and cfg.rotors_path is not None:
        raise ValueError(
            "both a RotorModel and --rotors were given; pass one. The file "
            "would be parsed and then discarded"
        )
    if rotors is None and cfg.rotors_path is not None:
        rotors = load_rotors(cfg.rotors_path)
        _log.info("rotors: %d from %s", len(rotors.spin), cfg.rotors_path)
    if rotors is not None and cfg.stub_physics:
        # Accepted and ignored: build_physics already has this semantics, and a
        # hard error would make --stub-physics awkward to use for transport
        # debugging on a vehicle that normally needs a sidecar.
        _log.warning("rotor parameters are ignored with --stub-physics")

    # Bind before loading the model: PX4 retries connect() every 500 us so
    # either start order works (plan 3.1), but binding first also means a port
    # collision is reported immediately rather than after MuJoCo has loaded.
    server = HilServer(cfg.hil_bind_host, cfg.hil_port)
    sidechannel = None
    try:
        physics = build_physics(cfg, rotors)
        if cfg.sidechannel_enabled:
            sidechannel = SideChannel(cfg.sidechannel_bind_host, cfg.sidechannel_port)
    except BaseException:
        # Nothing is running yet, but the listening socket is already bound and
        # would outlive us as a leaked fd, holding the port against a retry.
        if sidechannel is not None:
            sidechannel.close()
        server.close()
        raise

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
    try:
        cfg = config_from_args(argv)
    except (ValueError, FileNotFoundError) as exc:
        # A rejected flag combination or a missing model is user error, not a
        # crash: report it the way argparse reports a bad argument.
        print(f"{_PROG}: error: {exc}", file=sys.stderr)
        return 2
    _configure_logging(cfg.log_level)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
