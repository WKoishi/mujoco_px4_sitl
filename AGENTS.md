# Handoff

Start here when picking up this work. Current state, what is not built, what to do
next. No interface contract and no model authoring rules live here — see §5 for
which document owns what. §7 has the launch lines for each vehicle.

---

## 1. Where things stand

Phases 0–6 are complete and the quadrotor flies under PX4. `python -m pytest` is
green with no PX4 build and no GL.

`IMPLEMENTATION_PLAN.md` records the measurements taken when each phase was met —
Phase 3's attitude table exact to five decimals, Phase 5's full
arm → takeoff → 5 m square → land with its hover error table, Phase 6's side-channel
verification. **Those are the regression baseline. Do not copy them here, and do not
delete them from the plan when the platform changes.** If an X8 change breaks
something, the quad numbers are the only way to tell "the change broke it" from
"it never worked".

Phase 7 (aerial manipulator) is under way. Its first two prerequisites were done
earlier: `RotorModel` is ω-based with per-rotor arrays and an arbitrary rotor count
(`MODELING_CONVENTIONS.md` §2.5), and the PX4 side is consistent with it —
`THR_MDL_FAC 1.0`, `MPC_THR_HOVER 0.2025`, `MPC_THR_MIN 0.0144` (§2.6), those being
the **quad's** values. The conversion pipeline and the X8 model have since landed
(§3, §4).

**The quad has been flown twice on it**, once at `fac = 0` and once after the fix,
with hover error unchanged (0.153 m vertical / 0.121 m horizontal against ground
truth, inside the baseline's band). `IMPLEMENTATION_PLAN.md` phase 5 has both sets
of numbers and the two measurement traps.

`fac = 0` was not a tuning preference but an inconsistency: `CA_ROTOR*_CT` is
defined as `Thrust = CT · u²` while the effectiveness matrix is linear in the
actuator variable, so the allocator's variable is `u²` and the mixer's square root
is what converts it back. Without it the attitude loop ran at exactly 2.00× its
design gain — flyable, inside PX4's default margin, and invisible in a 5 m square
(fitting angular acceleration against commanded torque over that flight gives
R² ≈ 0.15, i.e. noise). The ratio is asserted on the plant instead, in
`tests/test_vehicle.py`.

**The X8 now reaches SITL.** The model is generated from the first CAD export by
`scripts/urdf_to_mjcf.py` and passes every self-check (§4); `--rotors` /
`MUJOCO_SITL_ROTORS` builds its 8-rotor `RotorModel` from the sidecar at startup;
and its PX4 airframe (`22002_mujoco_x8`, generated from
`px4/mujoco_x8.airframe.template`) carries the measured allocation. Verified at
load: 8 rotors, `MPC_THR_HOVER 0.2162`, and **no auto-calibration warning** —
which is the signal §4 describes, now silent because the measured values arrive.

**It has not flown.** Everything verified so far is on *our* side of the HIL
boundary: the model loads, `Vehicle` builds, the hover point is right. PX4's
allocator, mixer and attitude loop have not participated once.

Nothing about the layout itself is in doubt — the allocator is generic
(`NUM_ROTORS_MAX = 12`, and the effectiveness matrix is computed from `CA_ROTOR*`
as `ct·position×axis − ct·km·axis`), and PX4 ships `12001_octo_cox` as a coaxial
X8. The generated airframe was checked against it statically: pairs (0,5), (1,4),
(2,7), (3,6), `KM` opposed within each pair, every `PY` the negation of the
model's `py`. **What has never run is the closed loop** — those 32 generated
numbers, through PX4's matrix, against our MJCF's sites.

So three things are untested, all dynamic:

- **The index→ESC mapping**, assumed from `12001_octo_cox` and not confirmed
  against the wiring. When the allocator speeds up "rotor 0", whether the rotor
  that moves in MuJoCo is at the same place is not knowable statically. Presents
  as yaw drift or attitude cross-coupling.
- **Coaxial yaw authority.** X8 yaw comes from differential thrust within each
  pair, and `km` is 0.43× PX4's default. This channel does not exist on the quad,
  so no baseline covers it.
- **The arm's mass moving in flight.** No degree of freedom like it in Phase 5.

The 8-rotor coverage in `tests/test_vehicle.py` does not close this: it is a
synthetic inline model (a 3 kg box with 8 sites in `MODELING_CONVENTIONS.md` §4's
order), so it verifies force/torque algebra and the hover point, not flight.

The quad's own history is the argument for not trusting "it loads": the
`THR_MDL_FAC` inconsistency ran the attitude loop at 2.00× design gain and flew
anyway, invisible in a 5 m square. The X8 is at an earlier stage than that was.
Fly the Phase 5 profile (`scripts/fly_regression.py`) and compare against the
quad's baseline.

Motors are chosen: `c_t`, `km` and `ω_max` are fitted from the manufacturer's
data and live in the private sidecar (§4). `ω_idle` is not in that data and is
still a choice, and `vehicle.py`'s placeholders remain the fallback for any model
that does not supply its own — which is now only the quad.

---

## 2. What is not implemented

| Gap | Where | Consequence |
|---|---|---|
| `arm_cmd` never reaches the physics | `sidechannel.py` parses it into `ArmCommand`; nothing in `src/` writes `data.ctrl` | arm commands are received and silently dropped |
| **A dropped `arm_cmd` writer has no failsafe** | `ArmCommand` has no timestamp and no expiry | when the writer lands: a controller that crashes leaves the arm holding its last command forever. With under 8 cm to the nearest disc (§4 below), design the staleness behaviour with the writer, not after |
| Arm joint limits | exported as `0/0`; provisional values in the private sidecar | the arm moves, but over invented ranges |
| No arm state on the side channel | `SimState` has 9 fields, none of them joint state | an out-of-process controller is flying blind on the arm. Extending it touches `SimState`, the `Physics` protocol and `StubPhysics` |
| No aerodynamics | ω is available but nothing reads it | the effects `MODELING_CONVENTIONS.md` §5 lists are expressible, none are expressed |
| Coaxial `c_t` discount | all 8 rotors carry the isolated-rotor value | lower deck is modelled 15–25 % too strong (`MODELING_CONVENTIONS.md` §5 suggests 0.75–0.85). Deliberately deferred so the first X8 flight introduces one unverified quantity, not two: it moves `MPC_THR_HOVER` to 0.2377 / 0.2456 / 0.2541 at 0.85 / 0.80 / 0.75, so the sidecar's `c_t` and the airframe must change together |

One authoring hazard has **no test coverage in `src/`**: rotor sites must be
direct children of `base_link`, or the lever arm is silently wrong
(`MODELING_CONVENTIONS.md` §2.2). The conversion script now places them there by
construction and fails the check otherwise, so a generated model is safe — a
hand-written one is still exposed.

One more the `spin`-length check does **not** catch: a sidecar whose rotor count
matches the model but whose coefficients belong to a different vehicle. Both load
and fly, with the wrong thrust. Keeping the sidecar next to the MJCF it generated
is the only thing preventing it.

---

## 3. Next steps, in dependency order

Steps 1 and 2 — the ω-based `RotorModel` and the `MPC_THR_HOVER` / `THR_MDL_FAC`
recomputation — are **done and flight-verified on the quad** (§1;
`IMPLEMENTATION_PLAN.md` phase 5 has the numbers). `MODELING_CONVENTIONS.md` §2.5
records the parametrization and the re-derived `THR_MDL_FAC` argument;
`px4/22001_mujoco_quad` carries it as a comment block.

Step 3, the **conversion script and its sidecar schema**, is also done:
[scripts/urdf_to_mjcf.py](scripts/urdf_to_mjcf.py) plus
[models/x8_arm.conversion.yaml.template](models/x8_arm.conversion.yaml.template),
covered by `tests/test_urdf_to_mjcf.py`. It converts the first CAD export
end-to-end and the generated model passes every self-check (§4 has the numbers).

Steps 1 and 3 of the original list — building the `RotorModel` from the sidecar,
and the X8 airframe — are **done**: `--rotors` / `MUJOCO_SITL_ROTORS` with
`rotorconfig.py`, and `--emit-airframe` with `install_px4_files.sh --airframe`.
`RotorSpec` and the rotors-block parser moved into the package so the script and
the simulator share **one** parser; a second one would drift silently, since a
model whose rotor count still matches flies with the wrong thrust. What remains:

1. **Fly the X8.** Now the blocker, and it is a measurement rather than code. The
   vehicle loads with the right hover point but PX4's allocator has never seen
   this layout. `scripts/fly_regression.py` flies the Phase 5 profile; record the
   numbers in `IMPLEMENTATION_PLAN.md` phase 5 beside the quad's, and expect the
   index→ESC mapping to be what a failure implicates first (it presents as yaw
   drift, §4).
2. **Map `arm_cmd` onto `data.ctrl`**, scanning the `arm_act` prefix, in
   `step_frame` at [sim.py:137](src/mujoco_px4_sitl/sim.py#L137). `clear()` only
   zeros `base_link`'s `xfrc_applied`, so any writer that applies forces to the arm
   owns its own clearing. The generated model already has `arm_act0..4` with
   `ctrlrange` set. Decide the staleness behaviour here, not later (§2, row 2).
3. **Real arm joint limits** (§4 for why the export has none). The ranges in the
   private sidecar are invented.
4. **The coaxial `c_t` discount**, as its own step with its own flight, so it is
   separable from step 1's result.

Aerodynamic effects are deliberately **not** on this list. They come after Phase 7's
exit criterion; `RotorState.omega` is what makes them expressible.

### The research-controller boundary, decided once

The X8 is driven from a launcher in the private repo (`sim/run_x8.py`) that sets
`MUJOCO_SITL_ROTORS` and calls `run_sitl.sh`. So the research code is **out of
process** and sees only the side channel.

That choice buys isolation — a slow solver cannot stall lockstep — and sells two
things. Both are inherent to the process boundary, not schema gaps:

- **Determinism.** `serve()` polls and publishes without ever waiting, so the
  feedback delay is wall-clock and unreproducible. No counter reports it:
  `brake` and `timeouts` watch the PX4 side, not this one.
- **Speed.** `frame_wall_dt = imu_dt / speed_factor`, so at 5× the controller's
  unchanged wall-clock latency becomes 5× the sim-time delay. An out-of-process
  controller cannot batch experiments faster than real time.

Only an in-process controller called *synchronously inside* `step_frame` recovers
both, and it pays §6's price: a slow solver then shows up as PX4-side brake and
timeout counts. `run(cfg, rotors=...)` in `main.py` already accepts a built
`RotorModel` for that form, which is the hook and not the feature.

**So when the arm controller needs joint state, that is a topology decision, not
a request for JSON fields.** Adding `qpos` to `SimState` is easy and does not fix
either bullet above.

---

## 4. The platform: what is measured, and what is still input

**The first CAD export has landed** — a whole-vehicle SolidWorks export of the
coaxial X8 plus a 5-DoF arm, in the private repo (the URDF, its STLs, the filled
sidecar, the generated MJCF and **the generated PX4 airframe** all stay there;
this repo holds the script, the no-geometry template and the airframe template).
The airframe joins that list for the same reason as the rest: its `CA_ROTOR*_PX`
/ `_PY` block *is* the hub geometry. Naming already matched §3.3, so no renaming
was needed, and the STLs are in metres.

The conversion runs clean, and every self-check passes: mass and CoM match the
CAD figures, the 8 rotor sites are contiguous and on `base_link`, the coaxial
pairs' spins oppose, and the origin sits on the rotor symmetry centre to sub-mm
as §3.1 requires. **The numbers themselves live with the filled sidecar in the
private repo** — hub coordinates and an inertia tensor describe the airframe as
surely as the STLs do, which is the whole point of keeping the products there.
Run the script with `--check-only` to see them.

What the export did **not** carry:

- **Arm joint limits: every one is `lower=upper=0`.** The real blocker, and worse
  than it looks — MuJoCo reads 0/0 as *unlimited*, so the export's default is a
  free joint, not a locked one. The private sidecar has provisional ranges to
  keep work moving; they are invented and must be replaced. The zero-position and
  sign convention are needed with them, since `arm_cmd.values[i]` is an absolute
  angle.
- **Spin directions and index→ESC mapping.** The sidecar assumes
  `MODELING_CONVENTIONS.md` §4's PX4 order; it needs confirming against the
  wiring, since a mismatch presents as yaw drift.

**Motors are now chosen and measured**, which removes the largest remaining
guess. `c_t`, `km` and `ω_max` are fitted from the manufacturer's thrust/RPM/torque
table by least squares through the origin; the `c_t` fit is R² = 0.9997 with the
worst point 0.9 % off full thrust. Three consequences worth carrying forward:

- **Thrust/weight is 3.79, measured.** The auto-calibration's 4.0 was a
  construction; it is now within 5 % of it by coincidence, not by design.
- **`km` is 0.43× PX4's default.** Left at `CA_ROTOR*_KM 0.05` the allocator
  expects 2.3× the yaw torque these props make. Flyable, so it does not announce
  itself — it presents as sluggish yaw.
- **`MPC_THR_HOVER` moves to 0.2162**, from the quad's 0.2025. It depends only on
  `ω_idle / ω_max` and thrust-to-weight (§2.5), and `ω_max` is now a real number.

What is still missing from the motor side: `ω_idle`, because the datasheet starts
at 0.40 throttle, so the armed idle speed is not in it. The sidecar carries
`ω_max / 11` to match the quad's operating point, which is a choice rather than a
measurement, and `MPC_THR_HOVER` moves with it — 0.171 to 0.264 across the
plausible range. Bench-measure the armed idle RPM.

One clearance worth watching, and an argument for getting the limits right: the
gap between the arm's swept envelope and the nearest propeller disc is **under
8 cm**. Nothing in the model prevents a commanded angle from closing it — the
joint limits are the only thing that keeps the arm out of the propellers.

`vehicle.py`'s placeholders and the mass-based auto-calibration of `c_t` (§2.4)
**stay in place and still warn on every load** — they are what any model without
measured motor numbers falls back on, which is now only the quad. **The X8 loads
with no warning at all**, and that silence is the check: if it ever fires for the
X8, the sidecar's values did not reach `RotorModel`.

The generated airframe and the MJCF are both products of the same sidecar, which
is what keeps `CA_ROTOR*` and the rotor sites from disagreeing. Regenerate both
together; `--emit-airframe` runs after the self-check for that reason, so an
airframe is never derived from a model that failed its own mass check.

---

## 5. Which document wins

| Document | Owns | Wins on |
|---|---|---|
| `IMPLEMENTATION_PLAN.md` | PX4 v1.17 interface contract, phase design and exit criteria, troubleshooting, repo conventions, measured baselines | anything crossing the PX4 boundary: lockstep, message encoding, frames (§3.6 is the specification), geodetic origin |
| `MODELING_CONVENTIONS.md` | CAD → MJCF authoring, rotor parametrization, target platform, aerodynamic plans | anything about a model or a rotor: naming, frames *within* a model, thrust curve, X8 layout |
| `AGENTS.md` | state, gaps, next steps | "what now" |
| `README.md` | install, run, health check, side channel, ROS 2 integration | user-facing usage |

Where the first two overlap, the rule is scope, not seniority: §3.6's frame
conventions are the plan's and are binding on every model; the thrust curve is
`MODELING_CONVENTIONS.md` §2.5's and supersedes Phase 4's.

Progress belongs here, not in `README.md`. Measured results belong in the plan, at
the phase that produced them.

Two templates are documents in their own right, and own what no `.md` should
restate:

| Template | Owns |
|---|---|
| [models/x8_arm.conversion.yaml.template](models/x8_arm.conversion.yaml.template) | the sidecar schema, field by field, as comments |
| [px4/mujoco_x8.airframe.template](px4/mujoco_x8.airframe.template) | why a non-quad airframe is shaped the way it is; the boilerplate every generated one inherits |

Both are the no-geometry halves of files whose filled forms are private. Edit the
template, regenerate; never hand-edit a generated product.

---

## 6. Working conventions worth knowing up front

- **All repository text in English**, code comments included
  (`IMPLEMENTATION_PLAN.md` §8) — including documents drafted from a
  non-English conversation.
- **`source ../.venv/bin/activate` before any PX4 build.** PX4's cmake resolves its
  interpreter via `PATH` and configure aborts without it. See `README.md`.
- **No `rclpy` / `ament` / `ros` imports in `src/`.** The ROS 2 bridge is a separate
  package talking to the side channel over UDP.
- **Frame conversions live only in `frames.py`.** A rotation anywhere else is a bug.
- **`brake` and `timeouts` should both be 0 in healthy flight**, and `ratio` alone
  will not tell you otherwise. `IMPLEMENTATION_PLAN.md` §7 diagnoses both.
- **Commit messages: 10–20 lines of body.** The history runs to 53 lines and 640
  words because it restates derivations the documents already own. State a
  constraint once, in the document that owns it per §5, and have the commit point
  there. Worth a sentence in the commit and nowhere else: what an earlier version
  of the reasoning got wrong, since that is what a reader may already believe.

---

## 7. Running each vehicle

The quad needs nothing beyond `README.md`. The X8 spans two repositories, so its
launch line is worth stating once:

```sh
# Public repo, once per PX4 checkout:
./scripts/install_px4_files.sh --airframe ~/DEV/kani_arm/model/px4/22002_mujoco_x8
source ../.venv/bin/activate && make -C ../PX4-Autopilot px4_sitl_default

# Then, from the private repo:
python sim/run_x8.py            # sets MUJOCO_SITL_ROTORS, calls run_sitl.sh
```

Equivalently, by hand:

```sh
MUJOCO_SITL_ROTORS=~/DEV/kani_arm/model/mjcf/x8_arm.conversion.yaml \
PX4_SYS_AUTOSTART=22002 ./scripts/run_sitl.sh -m ~/DEV/kani_arm/model/mjcf/x8_arm.xml
```

**The sidecar is not optional and has no default.** Omitting it raises
`RotorModel.spin has 4 entries for 8 rotors` — deliberately, so there is no silent
fall back to the quad's placeholder motors. Conversely, the placeholder warning
firing on the X8 means the sidecar's values did not arrive.

Regenerating after a sidecar edit does both products at once, which is what keeps
`CA_ROTOR*` and the rotor sites from disagreeing:

```sh
python scripts/urdf_to_mjcf.py ~/DEV/kani_arm/model/mjcf/x8_arm.conversion.yaml \
    --emit-airframe ~/DEV/kani_arm/model/px4/22002_mujoco_x8
```

`--check-only` reports the measured numbers without writing anything, and is how
to read hub coordinates, mass and CoM without copying them into this repo.
