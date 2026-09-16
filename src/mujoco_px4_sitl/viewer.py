"""Optional MuJoCo viewer.

Off by default and deliberately decoupled: under lockstep the viewer's event
loop must not gate the physics loop, and headless CI must not need GL (plan
section 4). The viewer therefore only ever *reads* ``MjData`` and is synced at a
decimated rate from the physics loop's own thread.
"""

from __future__ import annotations

import logging
import time
from typing import Any

_log = logging.getLogger(__name__)


class Viewer:
    """Thin wrapper around ``mujoco.viewer.launch_passive``.

    Passive mode is the only correct choice here: it hands us the sync point
    instead of running its own loop, so the physics loop stays in charge.
    """

    def __init__(self, model: Any, data: Any, refresh_hz: float = 30.0) -> None:
        import mujoco.viewer  # imported lazily: needs GL

        self._handle = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        self._period = 1.0 / refresh_hz if refresh_hz > 0.0 else 0.0
        self._next = 0.0
        _log.info("viewer open (passive, %.0f Hz refresh)", refresh_hz)

    @property
    def alive(self) -> bool:
        return bool(self._handle.is_running())

    def sync(self) -> None:
        """Refresh at most every ``1/refresh_hz`` wall seconds."""
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + self._period
        self._handle.sync()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception as exc:  # noqa: BLE001 - closing must never propagate
            _log.debug("viewer close raised: %s", exc)
