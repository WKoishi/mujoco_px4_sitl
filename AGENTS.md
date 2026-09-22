# Handoff

Start here when picking up this work. Current state, what is not built, what to do
next. No interface contract and no model authoring rules live here — see §5 for
which document owns what.

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

**The X8 is a different matter: it cannot reach SITL yet**, though for a narrower
reason than before. The model now exists — generated from the first CAD export by
`scripts/urdf_to_mjcf.py`, passing every self-check (§4) — but nothing constructs
an 8-rotor `RotorModel` at startup, so loading it raises, and the PX4 airframe is
still `CA_ROTOR_COUNT 4`. The 8-rotor coverage in `tests/test_vehicle.py` is a
synthetic inline model (a 3 kg box with 8 sites in `MODELING_CONVENTIONS.md` §4's
order), so it verifies force/torque algebra and the hover point, not flight.

Motors are now chosen: `c_t`, `km` and `ω_max` are fitted from the manufacturer's
data and live in the private sidecar (§4). `ω_idle` is not in that data and is
still a choice, and `vehicle.py`'s placeholders remain the fallback for any model
that does not supply its own.

---

## 2. What is not implemented

| Gap | Where | Consequence |
|---|---|---|
| `arm_cmd` never reaches the physics | `sidechannel.py` parses it into `ArmCommand`; nothing in `src/` writes `data.ctrl` | arm commands are received and silently dropped |
| **Nothing builds a `RotorModel` from the sidecar** | `main.py:46` calls `build_physics(cfg)` with no `rotors` | **the X8 model cannot be run at all**: the default 4-entry `spin` raises against 8 rotor sites. The sidecar's measured `c_t` / `km` / `ω_max` are parsed, printed, and then dropped |
| Arm joint limits | exported as `0/0`; provisional values in the private sidecar | the arm moves, but over invented ranges |
| PX4 airframe is quad-only | [px4/22001_mujoco_quad](px4/22001_mujoco_quad) | `CA_ROTOR_COUNT 4`, no X8 allocation, `KM` at the 0.05 default |
| No aerodynamics | ω is available but nothing reads it | §5's effects are expressible, none are expressed |
| Coaxial `c_t` discount | all 8 rotors carry the isolated-rotor value | lower deck is modelled 15–25 % too strong (§5) |

One authoring hazard has **no test coverage in `src/`**: rotor sites must be
direct children of `base_link`, or the lever arm is silently wrong
(`MODELING_CONVENTIONS.md` §2.2). The conversion script now places them there by
construction and fails the check otherwise, so a generated model is safe — a
hand-written one is still exposed.

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
What remains:

1. **Build the `RotorModel` from the sidecar.** Now the blocker: the model exists
   and carries measured motor numbers, but `main.py` passes no `rotors`, so
   `RotorModel`'s quad default `spin` is rejected against 8 sites and the X8
   cannot be loaded at all. Everything below is downstream of this. The sidecar is
   a private file outside the repo, so the mechanism needs deciding — a `--rotors`
   path on the CLI, or having the conversion script emit the parameters into the
   MJCF as a `<custom><numeric>` block so the model is self-contained.
2. **Map `arm_cmd` onto `data.ctrl`**, scanning the `arm_act` prefix, in
   `step_frame` at [sim.py:127](src/mujoco_px4_sitl/sim.py#L127). `clear()` only
   zeros `base_link`'s `xfrc_applied`, so any writer that applies forces to the arm
   owns its own clearing. The generated model already has `arm_act0..4` with
   `ctrlrange` set.
3. **Switch the PX4 airframe to X8** (`MODELING_CONVENTIONS.md` §4 has the
   layout), with the measured numbers rather than defaults — `CA_ROTOR*_CT` 40.96,
   `KM` ±0.02154, `MPC_THR_HOVER` 0.2162. The conversion script prints all of
   them, and the private sidecar records the full parameter list. `THR_MDL_FAC`
   and `MPC_THR_MIN` are unchanged.
4. **Real arm joint limits** (§4 for why the export has none). The ranges in the
   private sidecar are invented.

Aerodynamic effects are deliberately **not** on this list. They come after Phase 7's
exit criterion; `RotorState.omega` is what makes them expressible.

---

## 4. The platform: what is measured, and what is still input

**The first CAD export has landed** — a whole-vehicle SolidWorks export of the
coaxial X8 plus a 5-DoF arm, in the private repo (the URDF, its STLs, the filled
sidecar and the generated MJCF all stay there; this repo holds only the script and
the no-geometry template). Naming already matched §3.3, so no renaming was needed,
and the STLs are in metres.

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
measured motor numbers falls back on, including the quad. Now that the X8's
numbers exist, the warning firing is a signal that the sidecar's values did not
reach `RotorModel`, which §2's first row says they currently cannot.

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
