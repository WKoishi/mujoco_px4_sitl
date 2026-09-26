"""The arm: joint commands into ``data.ctrl``, and propeller clearance.

Two jobs (AGENTS.md sections 1 and 4):

* :class:`ArmServos` maps ``arm_cmd`` onto the ``arm_act0..N`` position servos
  and owns what happens when commands stop arriving. That behaviour belongs to
  the simulated arm driver, so it runs on simulated time and is configurable:
  the research this simulator serves studies what the arm does when its
  controller stalls, so the simulator must not pick one answer silently.
* :class:`PropellerMonitor` measures how close the arm's collision capsules come
  to the propeller discs, every frame, on the pose the arm actually reached --
  not the commanded one, since the servo lags. It **reports and never blocks**:
  a simulator that clamped commands would hide exactly the violation the
  research has to show its controller avoids. A blocking guard belongs on the
  real vehicle (AGENTS.md section 4).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

ACTUATOR_PREFIX = "arm_act"

# What the arm driver does once the newest command is older than the timeout.
#   keep   -- servos hold the last target: a bus with no watchdog. Reported only.
#   freeze -- targets snap to the pose the joints have reached: "stop here".
#   limp   -- servo torque off: the arm falls under gravity and joint damping.
TIMEOUT_ACTIONS = ("keep", "freeze", "limp")

# ground_truth.time is rounded to the microsecond, so an honest echo of it can
# sit up to half a microsecond after the frame it was taken from.
_STATE_TIME_SLACK = 1e-6


def _log_this_one(count: int) -> bool:
    """Rate limit for per-message warnings: the first few, then every 100th.

    A controller streaming a malformed command at 50 Hz would otherwise bury
    the intrusion and staleness reports in the same log.
    """
    return count <= 3 or count % 100 == 0


@dataclass
class ArmCommand:
    """One ``arm_cmd`` as received. ``values[i]`` drives ``arm_act{i}``.

    ``state_time`` is optional: the ``ground_truth.time`` of the state the
    command was computed from. With it, a command's age is the age of the data
    behind it -- the controller's compute time and both transport legs
    included -- rather than the time since it arrived here.
    """

    seq: int = 0
    mode: str = "position"  # the only mode the generated actuators can honour
    values: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    state_time: float | None = None


@dataclass(frozen=True)
class JointReading:
    """Arm joint angles (rad) and rates (rad/s), one per ``arm_act*`` servo.

    Ideal encoders: MuJoCo's own joint state at the instant it was read, with no
    quantisation, noise or latency. The servo hardware that would set those is
    not chosen yet, so nothing here pretends to model it.
    """

    q: NDArray[np.float64]
    qd: NDArray[np.float64]


@dataclass
class ArmStatus:
    """A snapshot for the side channel and the status line."""

    cmd_seq: int | None
    # Simulated seconds since the command in force was computed (state_time) or,
    # without one, since it arrived. None before the first command.
    cmd_age: float | None
    cmd_stale: bool
    # Metres; negative means an arm capsule is inside a disc. None if unchecked.
    prop_clearance: float | None
    accepted: int = 0
    rejected: int = 0
    clamped: int = 0
    stale_episodes: int = 0
    intrusions: int = 0
    deepest: float = 0.0

    def summary(self) -> str:
        age = "-" if self.cmd_age is None else f"{self.cmd_age:.3f}s"
        clearance = (
            "unchecked" if self.prop_clearance is None
            else f"{self.prop_clearance * 1e3:+.1f}mm"
        )
        return (
            f"arm_cmd={self.accepted}{' STALE' if self.cmd_stale else ''} "
            f"age={age} stale_episodes={self.stale_episodes} "
            f"rejected={self.rejected} clamped={self.clamped} "
            f"prop_clearance={clearance} intrusions={self.intrusions}"
        )


class ArmServos:
    """Writes ``arm_cmd`` into ``data.ctrl`` and runs the command watchdog.

    Before the first command the servos hold the pose the model starts in.
    After it, the watchdog compares the command's age against ``timeout``
    (simulated seconds; 0 disables it) and applies ``on_timeout`` once when the
    age crosses it. The next command that is fresh on arrival resumes control;
    one that arrives already older than the timeout does not.
    """

    def __init__(self, model: mujoco.MjModel, timeout: float, on_timeout: str) -> None:
        if on_timeout not in TIMEOUT_ACTIONS:
            raise ValueError(
                f"arm timeout action {on_timeout!r} not in {TIMEOUT_ACTIONS}"
            )
        if not timeout >= 0.0:
            raise ValueError("arm timeout must be >= 0 (0 disables the watchdog)")
        self.model = model
        self.timeout = float(timeout)
        self.on_timeout = on_timeout

        ids: list[int] = []
        while True:
            actuator = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ACTUATOR_PREFIX}{len(ids)}"
            )
            if actuator < 0:
                break
            ids.append(actuator)
        self.ids = np.asarray(ids, dtype=np.intp)
        for index, actuator in enumerate(ids):
            self._require_position_servo(index, actuator)
        if model.nu != len(ids):
            # The scan stops at the first gap, so arm_act5 after a missing
            # arm_act4 would be silently undriven without this.
            _log.warning(
                "model has %d actuators but only %d contiguous %sN; the others "
                "are never written", model.nu, len(ids), ACTUATOR_PREFIX,
            )

        limited = np.asarray(model.actuator_ctrllimited[self.ids], dtype=bool)
        ranges = np.asarray(model.actuator_ctrlrange[self.ids], dtype=np.float64)
        self._low = np.where(limited, ranges[:, 0], -np.inf)
        self._high = np.where(limited, ranges[:, 1], np.inf)
        self._gainprm = np.array(model.actuator_gainprm[self.ids], dtype=np.float64)
        self._biasprm = np.array(model.actuator_biasprm[self.ids], dtype=np.float64)

        self.target = np.zeros(len(ids))
        self.command: ArmCommand | None = None
        self.received = math.nan
        self.stale = False
        self._stale_since = math.nan
        self.accepted = 0
        self.rejected = 0
        self.clamped = 0
        self.stale_episodes = 0

    def _require_position_servo(self, index: int, actuator: int) -> None:
        """``values[i]`` is an absolute joint angle, which only a unit-gear
        position servo on a joint honours (MODELING_CONVENTIONS.md section 3.4).
        A motor would take the angle as a torque and a geared servo would scale
        it, both silently."""
        model = self.model
        gain = float(model.actuator_gainprm[actuator, 0])
        bias = model.actuator_biasprm[actuator]
        is_servo = (
            int(model.actuator_gaintype[actuator]) == mujoco.mjtGain.mjGAIN_FIXED
            and int(model.actuator_biastype[actuator]) == mujoco.mjtBias.mjBIAS_AFFINE
            and int(model.actuator_trntype[actuator]) == mujoco.mjtTrn.mjTRN_JOINT
            and math.isclose(float(model.actuator_gear[actuator, 0]), 1.0)
            and gain > 0.0
            and abs(float(bias[0])) <= 1e-12
            and math.isclose(float(bias[1]), -gain, rel_tol=1e-9)
        )
        if not is_servo:
            raise ValueError(
                f"{ACTUATOR_PREFIX}{index} is not a unit-gear position servo on a "
                f"joint (gain kp, bias [0, -kp, -kv]); arm_cmd values are absolute "
                f"angles and only position servos are supported (AGENTS.md "
                f"section 2)"
            )

    @property
    def count(self) -> int:
        return len(self.ids)

    def hold(self, data: mujoco.MjData) -> None:
        """Target the pose the joints are in now. Call once after ``mj_forward``."""
        self.target = self._reached(data)
        data.ctrl[self.ids] = self.target

    def age(self, now: float) -> float | None:
        if self.command is None:
            return None
        start = self.received if self.command.state_time is None else self.command.state_time
        return max(0.0, now - start)

    # -- inbound ------------------------------------------------------------

    def submit(self, command: ArmCommand, now: float) -> bool:
        """Accept ``command`` as the one in force, or reject it and say why."""
        reason = self._reject_reason(command, now)
        if reason is not None:
            self.rejected += 1
            if _log_this_one(self.rejected):
                _log.warning("arm_cmd seq %d rejected (%d so far): %s",
                             command.seq, self.rejected, reason)
            return False
        values = np.asarray(command.values, dtype=np.float64)
        if np.any(values < self._low - 1e-9) or np.any(values > self._high + 1e-9):
            # Clamped rather than refused: the servo cannot go past its range any
            # more than MuJoCo's ctrlrange lets it. Counted so the controller
            # learns it asked for the impossible.
            self.clamped += 1
            if _log_this_one(self.clamped):
                _log.warning(
                    "arm_cmd seq %d outside ctrlrange, clamped (%d so far): %s",
                    command.seq, self.clamped, np.round(values, 4).tolist(),
                )
        self.command = command
        self.received = now
        self.accepted += 1
        return True

    def _reject_reason(self, command: ArmCommand, now: float) -> str | None:
        if self.count == 0:
            return f"the model has no {ACTUATOR_PREFIX}N actuators"
        if command.mode != "position":
            return (
                f"mode {command.mode!r}: the model has position servos only, so "
                f"no actuator can honour it (AGENTS.md section 2)"
            )
        values = np.asarray(command.values, dtype=np.float64)
        if values.shape != (self.count,):
            return f"{values.size} values for {self.count} arm actuators"
        if not np.all(np.isfinite(values)):
            return "non-finite value"
        if command.state_time is not None:
            if not math.isfinite(command.state_time):
                return "non-finite state_time"
            if command.state_time > now + _STATE_TIME_SLACK:
                return (
                    f"state_time {command.state_time:.6f} is after the simulator's "
                    f"{now:.6f}; echo ground_truth.time, not a wall clock"
                )
            held = self.command
            if held is not None and held.state_time is not None \
                    and command.state_time < held.state_time:
                return (
                    f"state_time {command.state_time:.6f} is older than the "
                    f"command in force ({held.state_time:.6f}); out of order"
                )
        return None

    # -- per frame ----------------------------------------------------------

    def update(self, data: mujoco.MjData) -> None:
        """Run the watchdog and write the targets. Call once per frame, before
        stepping, so a command received this frame drives this frame."""
        if self.count == 0:
            return
        if self.command is not None:
            age = self.age(float(data.time))
            stale = self.timeout > 0.0 and age is not None and age > self.timeout
            if stale and not self.stale:
                self._trip(data, age)
            elif self.stale and not stale:
                self._resume(data)
            self.stale = stale
            if not stale:
                self.target = np.clip(self.command.values, self._low, self._high)
        data.ctrl[self.ids] = self.target

    def _trip(self, data: mujoco.MjData, age: float) -> None:
        self.stale_episodes += 1
        self._stale_since = float(data.time)
        if self.on_timeout == "freeze":
            # The reached pose, not the last target. A gravity-loaded P servo
            # holds only with an error, so retargeting to where it sagged to
            # lets it sag by that much again -- as a real "goal = present
            # position" watchdog does.
            self.target = self._reached(data)
        elif self.on_timeout == "limp":
            self.model.actuator_gainprm[self.ids] = 0.0
            self.model.actuator_biasprm[self.ids] = 0.0
        _log.warning(
            "arm_cmd stale at t=%.3f s: newest command (seq %d) is %.3f s old, "
            "past --arm-timeout %.3f s; arm servos %s",
            float(data.time), self.command.seq if self.command else -1, age,
            self.timeout,
            {"keep": "keep the last target", "freeze": "freeze where they are",
             "limp": "go limp (torque off)"}[self.on_timeout],
        )

    def _resume(self, data: mujoco.MjData) -> None:
        if self.on_timeout == "limp":
            self.model.actuator_gainprm[self.ids] = self._gainprm
            self.model.actuator_biasprm[self.ids] = self._biasprm
        _log.warning(
            "arm_cmd resumed at t=%.3f s after %.3f s stale (seq %d)",
            float(data.time), float(data.time) - self._stale_since,
            self.command.seq if self.command else -1,
        )

    def joints(self, data: mujoco.MjData) -> JointReading:
        """The joints as they are, unclipped. Needs ``mj_forward`` current."""
        # With unit gear, actuator_length / _velocity are the joint's own.
        return JointReading(
            q=np.array(data.actuator_length[self.ids], dtype=np.float64),
            qd=np.array(data.actuator_velocity[self.ids], dtype=np.float64),
        )

    def _reached(self, data: mujoco.MjData) -> NDArray[np.float64]:
        # actuator_length is the joint angle itself, the gear being 1.
        reached = np.array(data.actuator_length[self.ids], dtype=np.float64)
        return np.clip(reached, self._low, self._high)


# --- propeller clearance --------------------------------------------------


def _point_disc_distance(
    p: NDArray[np.float64], c: NDArray[np.float64], n: NDArray[np.float64],
    radius: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Distance from points to flat discs, broadcasting over leading axes."""
    d = p - c
    z = np.einsum("...i,...i->...", d, n)
    in_plane = d - z[..., None] * n
    rho = np.sqrt(np.einsum("...i,...i->...", in_plane, in_plane))
    outside = np.maximum(rho - radius, 0.0)
    return np.sqrt(z * z + outside * outside)


def segment_disc_distance(
    a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64],
    n: NDArray[np.float64], radius: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact distance from each segment ``[a, b]`` to a flat disc.

    Row-wise over ``(P, 3)`` arrays; ``n`` must be unit normals, ``radius`` is
    ``(P,)``. Exact rather than sampled, because a sampled minimum is worst
    precisely where a segment grazes the rim, which is where an intrusion
    starts. See :class:`_SegmentsToDiscs` for how.
    """
    pairs = _SegmentsToDiscs(a, b, c, n, radius)
    pairs.refine(pairs.lower < pairs.upper)
    return pairs.upper


def least_clearance(
    a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64],
    n: NDArray[np.float64], radius: NDArray[np.float64],
    thickness: NDArray[np.float64],
) -> tuple[float, int]:
    """Exact ``min(distance - thickness)`` over the rows, and the row.

    What :class:`PropellerMonitor` needs each frame. Cheaper than taking the
    minimum of :func:`segment_disc_distance`: only rows whose lower bound could
    still beat the best closed-form candidate get the quartic -- usually one.
    """
    pairs = _SegmentsToDiscs(a, b, c, n, radius)
    best = float(np.min(pairs.upper - thickness))
    pairs.refine(pairs.lower - thickness < best)
    clearance = pairs.upper - thickness
    row = int(np.argmin(clearance))
    return float(clearance[row]), row


class _SegmentsToDiscs:
    """Segment-to-disc distances, row-wise, bracketed and then made exact.

    Distance to a convex set is convex along a segment, so the minimum sits at
    one of: a segment end; where the segment crosses the disc's plane; where it
    crosses the rim's cylinder; or a stationary point of the distance to the rim
    circle. The first three are closed-form and give ``upper``, which is already
    exact unless the minimum is of the fourth kind. ``lower`` bounds from below
    by the segment's least distance from the plane and from the rim's cylinder.

    Writing ``p(t) = a + t v``, the stationary points satisfy
    ``((p - c).v) rho = radius ((p - c)_plane . v_plane)``, which squares to a
    quartic in ``t``. :meth:`refine` solves it for the rows asked, in one
    batched eigenvalue call -- the expensive step, hence the bracket. Spurious
    roots are harmless: every candidate is evaluated and the least kept.
    """

    def __init__(
        self, a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64],
        n: NDArray[np.float64], radius: NDArray[np.float64],
    ) -> None:
        self.a, self.c, self.n, self.radius = a, c, n, radius
        self.v = v = b - a
        d0 = a - c
        z0 = np.einsum("ij,ij->i", d0, n)
        zv = np.einsum("ij,ij->i", v, n)
        w0 = d0 - z0[:, None] * n
        wv = v - zv[:, None] * n
        # In-plane distance from the axis, squared: s(t) = A t^2 + 2 B t + C.
        self.A = A = np.einsum("ij,ij->i", wv, wv)
        self.B = B = np.einsum("ij,ij->i", w0, wv)
        self.C = C = np.einsum("ij,ij->i", w0, w0)
        # (p(t) - c).v = L0 + V t.
        self.V = np.einsum("ij,ij->i", v, v)
        self.L0 = np.einsum("ij,ij->i", d0, v)
        R2 = radius * radius

        with np.errstate(divide="ignore", invalid="ignore"):
            plane = np.where(zv != 0.0, -z0 / zv, 0.0)
            disc = B * B - A * (C - R2)
            root = np.sqrt(np.maximum(disc, 0.0))
            crosses = (A > 0.0) & (disc >= 0.0)
            rim_in = np.where(crosses, (-B - root) / A, 0.0)
            rim_out = np.where(crosses, (-B + root) / A, 0.0)
            nearest_axis = np.where(A > 0.0, np.clip(-B / A, 0.0, 1.0), 0.0)
        t = np.stack([np.zeros_like(z0), np.ones_like(z0), plane, rim_in, rim_out], axis=1)
        self.upper = self._at(np.clip(t, 0.0, 1.0), slice(None))

        z1 = z0 + zv
        axial = np.where(z0 * z1 > 0.0, np.minimum(np.abs(z0), np.abs(z1)), 0.0)
        s_min = A * nearest_axis ** 2 + 2.0 * B * nearest_axis + C
        radial = np.maximum(np.sqrt(np.maximum(s_min, 0.0)) - radius, 0.0)
        self.lower = np.sqrt(axial * axial + radial * radial)

    def _at(self, t: NDArray[np.float64], rows: slice | NDArray[np.intp]) -> NDArray[np.float64]:
        """Least distance over parameters ``t`` (rows x candidates)."""
        points = self.a[rows, None, :] + t[:, :, None] * self.v[rows, None, :]
        return _point_disc_distance(
            points, self.c[rows, None, :], self.n[rows, None, :],
            self.radius[rows, None],
        ).min(axis=1)

    def refine(self, rows: NDArray[np.bool_]) -> None:
        """Make ``upper`` exact on ``rows`` by adding the quartic's roots."""
        index = np.flatnonzero(rows)
        if index.size == 0:
            return
        A, B, C = self.A[index], self.B[index], self.C[index]
        V, L0 = self.V[index], self.L0[index]
        R2 = self.radius[index] ** 2
        quartic = np.stack([
            V * V * A,
            2.0 * V * V * B + 2.0 * L0 * V * A,
            V * V * C + 4.0 * L0 * V * B + L0 * L0 * A - R2 * A * A,
            2.0 * L0 * V * C + 2.0 * L0 * L0 * B - 2.0 * R2 * A * B,
            L0 * L0 * C - R2 * B * B,
        ], axis=1)
        lead = quartic[:, 0]
        # A segment along the normal (A = 0) or of zero length (V = 0) has no
        # quartic; the closed-form candidates already cover it.
        proper = lead > 1e-12 * np.abs(quartic).max(axis=1)
        companion = np.zeros((index.size, 4, 4))
        companion[:, 1, 0] = companion[:, 2, 1] = companion[:, 3, 2] = 1.0
        companion[:, :, 3] = -quartic[:, :0:-1] / np.where(proper, lead, 1.0)[:, None]
        companion[~proper] = 0.0
        t = np.clip(np.linalg.eigvals(companion).real, 0.0, 1.0)
        self.upper[index] = np.minimum(self.upper[index], self._at(t, index))


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        body = int(model.body_parentid[body])
        if body == ancestor:
            return True
    return False


class PropellerMonitor:
    """Clearance between the arm's collision capsules and the propeller discs.

    A disc is a rotor site of cylinder type: flat, in the site's xy-plane, with
    radius ``size[0]``. ``size[1]`` is drawing thickness only. The arm is every
    body below ``base_link``; its capsules and spheres are the collision geoms
    checked. Clearance is the distance from a capsule's axis to the disc minus
    the capsule radius, so it bottoms out at minus that radius once the axis
    itself crosses the disc.

    Only reports: a count, the deepest penetration, and one log line when each
    intrusion starts and ends. Nothing here touches ``data``.
    """

    def __init__(
        self, model: mujoco.MjModel, base_id: int, rotor_site_ids: list[int]
    ) -> None:
        self.model = model
        cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        discs = [s for s in rotor_site_ids if int(model.site_type[s]) == cylinder]
        arm_bodies = {
            body for body in range(1, model.nbody)
            if body != base_id and _descends_from(model, body, base_id)
        }
        round_types = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE))
        capsules: list[int] = []
        unchecked: list[int] = []
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in arm_bodies:
                continue
            if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
                continue  # visual only
            (capsules if int(model.geom_type[geom]) in round_types else unchecked).append(geom)

        if arm_bodies and not capsules:
            _log.warning(
                "the arm has no collision capsules: propeller intrusion is NOT "
                "checked"
            )
        elif capsules and not discs:
            _log.warning(
                "no rotor site carries a propeller radius (a cylinder site), so "
                "propeller intrusion is NOT checked. Give each rotor a radius in "
                "the conversion sidecar and reconvert"
            )
        elif capsules and len(discs) < len(rotor_site_ids):
            _log.warning(
                "only %d of %d rotor sites are discs; the others are not checked "
                "for propeller intrusion", len(discs), len(rotor_site_ids),
            )
        if unchecked and discs:
            _log.warning(
                "arm collision geoms %s are not capsules or spheres and are not "
                "checked against the propeller discs",
                ", ".join(self._name(mujoco.mjtObj.mjOBJ_GEOM, g) for g in unchecked),
            )

        self.enabled = bool(capsules and discs)
        self._capsules = np.asarray(capsules, dtype=np.intp)
        self._discs = np.asarray(discs, dtype=np.intp)
        sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        self._half = np.array([
            0.0 if int(model.geom_type[g]) == sphere else float(model.geom_size[g, 1])
            for g in capsules
        ])
        self._thickness = np.array([float(model.geom_size[g, 0]) for g in capsules])
        self._disc_radius = np.array([float(model.site_size[s, 0]) for s in discs])
        # Every capsule against every disc, flattened.
        self._pc = np.repeat(np.arange(len(capsules)), len(discs))
        self._pd = np.tile(np.arange(len(discs)), len(capsules))

        self.clearance: float | None = None
        self.intrusions = 0
        self.deepest = 0.0
        self._inside = False
        self._since = 0.0
        self._episode_depth = 0.0
        self._episode_pair = 0
        if self.enabled:
            _log.info(
                "propeller clearance: %d arm capsule(s) against %d disc(s), R = %s m",
                len(capsules), len(discs),
                "/".join(sorted({f"{r:.4f}" for r in self._disc_radius})),
            )

    def _name(self, kind: mujoco.mjtObj, index: int) -> str:
        return mujoco.mj_id2name(self.model, kind, int(index)) or f"#{index}"

    def pair_names(self, pair: int) -> tuple[str, str]:
        """``(capsule geom, rotor site)`` for a pair index from :meth:`measure`."""
        geom = self._capsules[self._pc[pair]]
        site = self._discs[self._pd[pair]]
        return (self._name(mujoco.mjtObj.mjOBJ_GEOM, geom),
                self._name(mujoco.mjtObj.mjOBJ_SITE, site))

    def check(self, data: mujoco.MjData) -> None:
        """Measure the reached pose and track intrusions. Call once per frame,
        with kinematics current (after ``mj_forward``)."""
        if not self.enabled:
            return
        clearance, pair = self.measure(data)
        self.clearance = clearance
        self._track(float(data.time), -clearance, pair)

    def measure(self, data: mujoco.MjData) -> tuple[float, int]:
        """Least clearance over every capsule/disc pair, and which pair.

        No bookkeeping, so a survey over many poses can call it too. Needs
        ``geom_x*`` and ``site_x*`` current -- ``mj_kinematics`` is enough.
        """
        axis = data.geom_xmat[self._capsules].reshape(-1, 3, 3)[:, :, 2]
        centre = data.geom_xpos[self._capsules]
        a = centre - self._half[:, None] * axis
        b = centre + self._half[:, None] * axis
        disc_centre = data.site_xpos[self._discs]
        normal = data.site_xmat[self._discs].reshape(-1, 3, 3)[:, :, 2]
        return least_clearance(
            a[self._pc], b[self._pc], disc_centre[self._pd], normal[self._pd],
            self._disc_radius[self._pd], self._thickness[self._pc],
        )

    def _track(self, now: float, depth: float, pair: int) -> None:
        if depth > 0.0:
            self.deepest = max(self.deepest, depth)
            if not self._inside:
                self._inside = True
                self.intrusions += 1
                self._since = now
                self._episode_depth = depth
                self._episode_pair = pair
                if _log_this_one(self.intrusions):
                    geom, site = self.pair_names(pair)
                    _log.warning(
                        "propeller intrusion #%d at t=%.3f s: %s is %.1f mm "
                        "inside %s's disc (reported, not blocked)",
                        self.intrusions, now, geom, depth * 1e3, site,
                    )
            elif depth > self._episode_depth:
                self._episode_depth = depth
                self._episode_pair = pair
        elif self._inside:
            self._inside = False
            if _log_this_one(self.intrusions):
                geom, site = self.pair_names(self._episode_pair)
                _log.warning(
                    "propeller intrusion #%d ended at t=%.3f s after %.3f s; "
                    "deepest %.1f mm (%s in %s's disc)",
                    self.intrusions, now, now - self._since,
                    self._episode_depth * 1e3, geom, site,
                )
