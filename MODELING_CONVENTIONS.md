# Model authoring: SolidWorks → URDF → MJCF

Where the Phase 7 (aerial manipulator) model comes from. `IMPLEMENTATION_PLAN.md`
is the interface contract; this file covers **how the model is built** — and wins
within that scope, superseding the plan's Phase 4 thrust curve and Phase 7 model
details. The plan's §3.6 frame conventions are binding on every model here.
`AGENTS.md` §5 has the full split.

Target platform: **coaxial X8 octorotor + serial arm**, whole vehicle exported
from SolidWorks.

---

## 1. Why MJCF is mandatory

MuJoCo can read URDF, but URDF has no syntax for three things this simulator
requires:

| Missing | Purpose | Code that depends on it |
|---|---|---|
| `<site>` | rotor thrust application points `rotor0..7`, IMU mount `imu` | [vehicle.py:73-83](src/mujoco_px4_sitl/vehicle.py#L73-L83) |
| `<sensor>` | `imu_accel` / `imu_gyro` | [sim.py:70-73](src/mujoco_px4_sitl/sim.py#L70-L73) |
| `freejoint` | the base's 6 DoF | [sim.py:76-88](src/mujoco_px4_sitl/sim.py#L76-L88) |

One more: MuJoCo's URDF parser **welds the root link to the world**, so the
`freejoint` has to be added by hand regardless.

So: **URDF is only a courier for the CAD** (geometry, inertia, kinematic tree).
MJCF is the model source file. Once converted, the URDF plays no further part in
the simulation.

---

## 2. How the simulator identifies the actuated parts

### 2.1 Rotors: by site name, not by MJCF actuator

Thrust and reaction torque are computed in Python and written straight into
`base_link`'s `xfrc_applied`
([vehicle.py:147-166](src/mujoco_px4_sitl/vehicle.py#L147-L166)). The rules:

- Site names `rotor0`, `rotor1`, … are scanned, and **the scan stops at the first
  index that is missing**. A gap (`rotor0,rotor1,rotor3`) is read as a 2-rotor
  vehicle, with no error.
- `HIL_ACTUATOR_CONTROLS.controls[i]` drives `rotor{i}`; PX4 numbers from 0.
- Thrust model `T_i = c_t_i · ω_i²`, with ω an explicit state and a first-order
  motor lag of `time_constant = 0.02 s` on the command. §2.5 is the
  parametrization.
- Reaction torque `= km · T`, signed by `spin[i]` (+1 = CCW about body +z).

**Rotors are not mechanical degrees of freedom.** No body, no joint, no
actuator — `rotor{i}` is purely a marker for where a force is applied. Blade
geometry belongs to `base_link` as fixed geoms.

### 2.2 Two ways to fail silently

**Rotor sites must be direct children of `base_link`.**
[vehicle.py:115-119](src/mujoco_px4_sitl/vehicle.py#L115-L119) subtracts
`base_link`'s `body_ipos` from `site_pos` directly, but `site_pos` is local to
**whichever body owns the site**. A site on a child body raises no error, gets the
wrong lever arm, and presents as attitude cross-coupling. **No test covers this.**

**`spin` must have exactly one entry per rotor.** A mismatch is a hard error at
[vehicle.py:130-136](src/mujoco_px4_sitl/vehicle.py#L130-L136) — it is not
truncated, because flying an X8 on the default 4-entry quad tuple would present
as yaw drift. `spin` is geometry and cannot be inferred, so its length is what
declares the expected rotor count; `build_physics` takes the `RotorModel` that
carries it.

### 2.3 The arm: commands never reach the physics

The side channel parses `arm_cmd` into `ArmCommand(mode, values)`
([sidechannel.py:95-103](src/mujoco_px4_sitl/sidechannel.py#L95-L103)), but
**nothing in `src/` writes `data.ctrl`** — the command is received and dropped.
Phase 7 work.

### 2.4 Auto-calibrated `c_t` is the fallback when motors are unchosen

`c_t = None` calibrates from `body_subtreemass` so that full command gives
`thrust_to_weight` (default 4.0), and **warns every time**
([vehicle.py:186-210](src/mujoco_px4_sitl/vehicle.py#L186-L210)), with a second
warning if the result is more than 3× off `CT_PLACEHOLDER`. Upside: any arm mass
still flies, and hover stays at the same command whatever the vehicle weighs — one
`MPC_THR_HOVER` covers quad, X8 and arm. Downside: **the motors quietly get
stronger**, hiding a real platform that would be underpowered. That is what the
warning is for.

Dropping it outright would turn "heavy arm" into "will not take off", which is
correct physics but presents as a broken model. Once the motors are chosen, pass a
measured `c_t` and the calibration is bypassed.

### 2.5 ω as an explicit state

**Implemented.** `k_thrust` folded `C_T` and `ω_max²` into one scalar that could
not be taken apart again. **Rotor speed is what every aerodynamic effect in §5
needs** — advance ratio, flapping, coaxial interference and gyroscopic precession
all take ω, not a normalized command. `RotorModel` is now:

```
ω_i   = clip(ω_idle + (ω_max_i − ω_idle) · u_i,  ω_min_i, ω_max_i)
T_i   = c_t_i · ω_i²
τ_z_i = −spin_i · km_i · T_i
```

Adopted from PegasusSimulator's `QuadraticThrustCurve`
(`logic/thrusters/quadratic_thrust_curve.py`), which takes the same shape. What is
taken and what is not:

| Adopt | Why |
|---|---|
| ω as state, exposed to the aero layer | the prerequisite for everything in §5 |
| Per-rotor arrays `c_t`, `km`, `spin`, `ω_min`, `ω_max` | answers the coaxial discount in §5; also unblocks X8 (§2.2) |
| `c_t` / `ω_max` from motor+prop data | `k_thrust` is a lumped fit with no datasheet counterpart |
| Non-zero idle speed | a real motor does not stop at zero command when armed |

| Reject | Why |
|---|---|
| Their instantaneous model (`no delay introduced`) | keep our `τ = 0.02 s` lag; motor response affects the attitude loop's phase margin |
| Their decoupled `τ_z = Σ k_m ω² rot_dir` | algebraically equivalent (`k_m = km · c_t`), but `τ = km · T` **is** PX4's `CA_ROTOR*_KM` definition (`control_allocator/module.yaml:211-218`). Worth noting their defaults imply `k_m / C_T = 0.117` against PX4's default `KM` of 0.05 — a 2.3× disagreement, which is the argument for staying pinned to PX4's definition |
| `update(state, dt)` with both arguments unused | do not add dead parameters |

Because ω is affine in `u`, low-passing `u` and low-passing ω are equivalent (the
constant idle term is unaffected by the filter), so the existing lag implementation
carries over unchanged.

Coefficients are scalar-or-per-rotor: a scalar broadcasts, a sequence must match
the rotor count. `ω_idle` is shared — it is an ESC setting, not a per-rotor one.

**The one real knock-on cost.** A non-zero idle speed moves the hover operating
point. With `ω_idle / ω_max = 1/11` and full-command thrust-to-weight at 4.0:

| | Old `k_thrust · u²` | ω-based |
|---|---|---|
| Hover command | 0.500 | **0.450** |
| Thrust at zero command | 0 | 3.3 % of weight |

(An earlier version of this table compared local gains against a linear plant of
the same full-command thrust, giving 4.000 against 3.636 — "0.909×". The figure is
arithmetically right but answers nothing useful: the reference is full scale, not
any model PX4 holds, and it invited being read as a closed-loop regression. §2.6
has the comparison that matters.)

Note this is mass-independent: the hover command depends only on `ω_idle / ω_max`
and `thrust_to_weight`, so 0.450 covers the quad, the X8 and any arm mass on top —
which is what makes a single `MPC_THR_HOVER` viable at all. Verified in
`tests/test_vehicle.py` against both a 1.52 kg quad and a 3 kg X8.

**`MPC_THR_HOVER` moves with the idle offset.** Hover is at `u = 0.450`, so the
parameter is 0.450 at `fac = 0` and `0.450^2 = 0.2025` at `fac = 1` -- it lives in
the allocator's domain, which section 2.6 shows is `u^2`. Measured from a ulog:
`actuator_motors` sits at `MPC_THR_HOVER` in hover, 1:1, which is what fixes the
relationship.

`THR_MDL_FAC` is **1**, and section 2.6 owns that argument in full. Superseded
here: earlier versions of this section argued for leaving it at 0, on the grounds
that the plant and PX4's linear allocator shared a slope at mid-stick and the idle
offset only spoiled that by 0.909x. Both the framing and the conclusion were
wrong. `fac` is not a tuning knob for slope agreement -- it is what makes the
mixer consistent with `CA_ROTOR*_CT`'s own definition, and at 0 the attitude loop
runs at exactly twice its design gain.

### 2.6 `THR_MDL_FAC` must be 1, and why

**Fixed.** `px4/22001_mujoco_quad` sets `THR_MDL_FAC 1.0`,
`MPC_THR_HOVER 0.2025`, `MPC_THR_MIN 0.0144`. These are one setting; changing any
one alone leaves the vehicle flyable but mistuned.

This is a consistency requirement of PX4's own rotor model, not a linearization
of ours. `CA_ROTOR*_CT` is defined as **`Thrust = CT · u²`**
(`control_allocator/module.yaml:196-209`), while the effectiveness matrix is
**linear** in the actuator variable — `ct * axis` and `ct * position.cross(axis)`
(`ActuatorEffectivenessRotors.cpp:195-198`). Both can only hold if the
allocator's actuator variable is `u²`, and `FunctionMotors`' square root at
`fac = 1` is what converts it back to `u`
(`mixer_module/functions/FunctionMotors.hpp:81-107`).

At `fac = 0` that step is the identity, so the plant's square stays in the loop:

```
control_allocator   thrust domain, linear, correct
      |             actuator_motors[i]
FunctionMotors      fac = 1 -> sqrt          fac = 0 -> identity
      |             u = sqrt(s)              u = s
our plant           T = c_t * omega(u)^2     T = c_t * omega(u)^2
                    torque matches intent    torque is 2.00x intent
```

**The factor is exactly 2.00**, and it does not depend on the idle offset.
Perturbing an allocation by ε about hover command `h`, with
`omega_h = omega_idle + k·h`:

| | `omega` | `ΔT` to first order |
|---|---|---|
| `fac = 0`, `s = h(1+ε)` | `omega_idle + k·h(1+ε)` | `2·c_t·omega_h·k·h·ε` |
| `fac = 1`, `s = h²(1+ε)` | `omega_idle + k·h·sqrt(1+ε)` | `c_t·omega_h·k·h·ε` |

Both share `omega_h`, so the ratio is 2 regardless of `omega_idle` — it is just
the factor differentiating a square introduces. So **`fac = 0` ran the attitude
loop at exactly twice its design gain.** It flew anyway, inside PX4's default
gain margin, which is why this needed measuring rather than trusting.

`MPC_THR_MIN` is the trap. It is a floor on the *setpoint*, so at `fac = 1` the
default 0.12 clamps the command at `sqrt(0.12) = 0.346` — 0.66 of vehicle weight,
and the vehicle cannot descend. `0.12² = 0.0144` restores the same command floor
and with it the identical physical range, T/W 0.160 to 4.000.

**There is no cross-axis leak, at either setting.** An earlier draft of this
section claimed a commanded pure yaw leaks roll and pitch. It does not: the
quadratic term a command-domain split adds is identical at all four `(x, y)`
positions, so it cancels by symmetry. That draft generalized from a test that
drove a *single* coaxial pair, which is asymmetric by construction and not what
the allocator emits. The error was only ever a gain scale.
`test_no_cross_axis_leak_at_either_fac` pins this down so the wrong version does
not come back.

Measured on the plant rather than in flight, deliberately: a 5 m square is far
too gentle to excite the attitude loop enough to see a 2× gain in a ulog. Fitting
angular acceleration against commanded torque over the regression flight gives
R² ≈ 0.15 — noise, not signal. The flight's job is confirming no regression
(`IMPLEMENTATION_PLAN.md` phase 5); the ratio's job belongs to
`test_thr_mdl_fac_1_is_what_makes_torque_match_the_allocation`.

---

## 3. Conventions to settle before exporting

### 3.1 Coordinate frame (the critical one)

Create a Reference Coordinate System in the assembly: **X = forward, Y = left,
Z = up**, and select it as `base_link`'s frame on export. Put the origin at the
airframe's geometric centre (the rotor symmetry centre), not at the CAD modelling
datum.

Getting this wrong corrupts PX4's attitude, position and body rates. It presents
as slow EKF divergence, which is hard to trace back to the model.

The world frame is ENU (+x east, +y north). **A MuJoCo identity attitude is PX4
yaw +90°**, `yaw_px4 = 90° − yaw_mujoco`. That is expected, not a bug — see §3.6
of the plan.

### 3.2 Splitting into links

**The entire rigid airframe must be one link.** Frame, arm tubes, rotor hubs,
blades, landing gear, flight controller and battery all go into `base_link`. Only
the manipulator's moving segments are separate links. Reason: §2.2.

**Do not give the propellers rotational mates.** Exported as revolute joints they
add 8 undriven degrees of freedom that only cost solver time and add numerical
noise.

**Keep blades out of collision.** The convex hull of 8 blades wraps the whole
vehicle, so takeoff, landing and arm motion all hit the ground. Visual uses the
CAD mesh; collision uses hand-written primitives.

### 3.3 Naming

| Object | Name | Hardcoded |
|---|---|---|
| base body | `base_link` | yes |
| rotor sites | `rotor0` … `rotor7` | yes (prefix + 0-based, contiguous) |
| IMU site | `imu`, `quat="1 0 0 0"` | yes |
| IMU sensors | `imu_accel` / `imu_gyro` | yes |
| arm joints | `arm_joint0` … | no |
| arm links | `arm_link0` … | no |
| arm actuators | `arm_act0` … | no |

The arm is 0-based to match the rotors and PX4.

### 3.4 Arm joints

- All **`revolute`**, never `continuous` (a position actuator needs `ctrlrange`).
- Decide each joint's angle limits in SolidWorks.
- Ordered from the base outward.
- In MJCF they become `<position kp=… ctrlrange=…>`. Joints need `damping` and
  `armature` (gearbox friction and reflected inertia) — that is what keeps a stiff
  position actuator numerically stable.
- Side-channel semantics: `arm_cmd.values[i]` is `arm_joint{i}`'s **absolute angle
  in radians**.

### 3.5 Mass properties and meshes

- **Every part needs a material or density.** `<inertial>` comes straight from
  SolidWorks' mass properties, which is the main reason to go through CAD at all.
  Blades included — their mass is real.
- Confirm the exported STL is in metres. CAD commonly exports millimetres, which
  needs `meshscale="0.001"`.
- SolidWorks STL exports run to hundreds of thousands of faces. Decimate them.
- Layout `models/<name>/` plus `meshes/`, with `<compiler meshdir="meshes">`.
  MuJoCo does not understand `package://` URIs; the conversion script rewrites
  them.
- `timestep` in the XML is overridden by `--physics-rate`
  ([sim.py:63](src/mujoco_px4_sitl/sim.py#L63)). Setting it has no effect.

---

## 4. Coaxial X8 layout

PX4 ships [`12001_octo_cox`](../PX4-Autopilot/ROMFS/px4fmu_common/init.d/airframes/12001_octo_cox)
as its coaxial X8 reference. Converted to FLU (negate y and z), with unwritten
`KM` taking the default `+0.05`:

| idx | Position (FLU) | Deck | Spin |
|---|---|---|---|
| 0 | front right (+0.35, −0.35) | upper | CCW |
| 1 | front left (+0.35, +0.35) | upper | CW |
| 2 | rear left (−0.35, +0.35) | upper | CCW |
| 3 | rear right (−0.35, −0.35) | upper | CW |
| 4 | front left (+0.35, +0.35) | lower | CCW |
| 5 | front right (+0.35, −0.35) | lower | CW |
| 6 | rear right (−0.35, −0.35) | lower | CCW |
| 7 | rear left (−0.35, +0.35) | lower | CW |

**The coaxial pairs are (0,5), (1,4), (2,7), (3,6)** — not (0,4), (1,5). The lower
deck's order is the upper deck's mirror with the pairs swapped, which is what lets
`KM` alternate sign cleanly along indices 0…7. This is easy to wire up wrong; copy
PX4's order rather than inventing one, so a misbehaving model can be compared
against a known-good official airframe.

`spin` is therefore `(+1, -1, +1, -1, +1, -1, +1, -1)`.

**`PZ` produces no torque.** The height difference between the upper and lower
rotors contributes **nothing** to roll or pitch in this thrust model — thrust is
along body +z, and `r × [0,0,T]` only uses `r`'s x and y components. PX4's
allocator behaves the same way. `CA_ROTOR*_PZ` and the site z coordinate are
documentation only.

**A coaxial pair's `KM` values have opposite signs**, so driving one pair's upper
and lower rotors differentially produces pure yaw torque with no roll or pitch. An
X8 has far more yaw authority than a quad of the same size.

### PX4 airframe changes

[px4/22001_mujoco_quad](px4/22001_mujoco_quad): set `CA_ROTOR_COUNT` to 8, fill in
all 8 `CA_ROTOR*` blocks, `PWM_MAIN_FUNC1..8 = 101..108`, add `MAV_TYPE 14`, and
change the filename and `@type` to `Octorotor Coaxial`.

No MAVLink-side change: `HIL_ACTUATOR_CONTROLS` carries 16 channels
([hil.py:32](src/mujoco_px4_sitl/hil.py#L32)), and `effective(count)` takes the
first N.

---

## 5. Should blades be separate links?

**No — it buys nothing aerodynamically and the cost is prohibitive.** Two measured
results (MuJoCo 3.13.0, this workspace):

**MuJoCo has no blade-element aerodynamics.** Axial fluid force on two flat-plate
blades (0.2 m diameter) at 600 rad/s (≈5700 RPM):

| Fluid model | 0° pitch | 15° pitch |
|---|---|---|
| inertia-box (`quad_x.xml` default) | +0.0045 N | +0.0041 N |
| ellipsoid | +2.35 N | +0.54 N |

The default model produces essentially no thrust (0.0045 N against 9.87 N of
weight). The ellipsoid model does produce a force, but **its pitch dependence runs
backwards** — 0° exceeds 15°, where a real rotor at zero pitch produces zero
thrust. That is bluff-body drag, not blade lift. So even as separate links, the
thrust still has to be computed analytically, and the extra degrees of freedom are
pure overhead.

**The timestep cost settles it.** `quad_x.xml` measures 74k steps/s
(13.5 µs/step), so a 250 Hz IMU frame has a 296-step budget at real time.
Resolving a 5700 RPM blade to 5°/step needs roughly **36 kHz** — 36× the current
1 kHz, which consumes the entire real-time margin single-threaded, before the arm's
contact solve and 8 hinges. Under lockstep that is not "somewhat slower": it is the
ratio collapsing to 0.03 and PX4's clock stalling.

### What aerodynamics actually needs: rotor-disc inflow

The rotor model is still pure feedforward — thrust is a function of command only,
and `vehicle.py` **reads no velocity state at all**. None of the effects below need
separate links, but all of them need rotor speed, which §2.5 now supplies as
`RotorState.omega`:

| Effect | What it needs |
|---|---|
| Ground effect | disc height above ground + rotor radius |
| Flapping / H-force | freestream velocity at the disc + ω (advance ratio) |
| Coaxial interference | upper-rotor induced velocity → lower-rotor `c_t` discount |
| Forward-flight thrust loss | body velocity projected onto the disc + ω |
| Downwash onto the arm | momentum-theory wake model + `xfrc_applied` |

The one effect that *would* come naturally from a separate link is **gyroscopic
precession**, and even that is analytic: apply `ω_body × (I_prop · ω)`, with ω taken
from the rotor state of §2.5. A few lines, zero extra DoF.

This is also what RotorS and gz-sim's `MulticopterMotorModel` do — rotors as force
application points plus analytic aerodynamics, never actually spinning.

### Aerodynamic parameters to record in CAD

Blades are modelled as fixed geoms on `base_link` (with mass, no contact, no
joint), but record:

```
rotor radius R            ← ground effect, induced velocity, wake
blade count               ← solidity
blade pitch (geometric)   ← future blade-element approximation
upper / lower disc height ← coaxial spacing
I_prop about the spin axis ← future gyroscopic term; SolidWorks gives it directly
arm workspace vs. the disc ← downwash
```

From the motor and propeller datasheets rather than CAD, for §2.5:

```
c_t per rotor   [N/(rad/s)²]  ← thrust coefficient, from the measured thrust curve
ω_max per rotor [rad/s]       ← at the flight battery voltage, not the nominal KV
ω_idle          [rad/s]       ← armed-but-idle speed
```

A free option worth taking: **write an explicit `zaxis` on each `rotor*` site**.
The code currently hardcodes thrust along body +z and ignores site orientation, but
with `zaxis` in place, supporting tilt-rotors or a tilted disc is a few lines
later. It costs nothing now.

### Known fidelity gaps (recorded deliberately, not oversights)

- Blade rotational inertia and gyroscopic effects are not modelled. Fast yaw
  manoeuvres will be missing the precession torque. §2.5 supplies the ω this needs.
- Coaxial aerodynamic interference is not modelled. The lower rotor sits in the
  upper rotor's downwash and typically makes only 75–85 % of an isolated rotor's
  thrust; the current model treats all 8 as identical. Auto-calibration still puts
  total hover thrust at command 0.5, but the upper/lower asymmetry is lost. **The
  per-rotor `c_t` array of §2.5 is the mechanism; the discount value is still
  undecided (§7).**

---

## 6. Conversion pipeline

**Implemented**: [scripts/urdf_to_mjcf.py](scripts/urdf_to_mjcf.py), with
[models/x8_arm.conversion.yaml.template](models/x8_arm.conversion.yaml.template)
as the sidecar schema and `tests/test_urdf_to_mjcf.py` covering it.

**The CAD is private, and so are the conversion products.** The URDF, its STLs,
the filled sidecar and the generated MJCF all live in the private repo; this
repository holds the script and the no-geometry template only. Rotor hub
coordinates and an inertia tensor describe the airframe as surely as a mesh does,
so a filled sidecar is not a configuration file that can be published. Nothing in
`models/` but `quad_x.xml` and the template.

The CAD will be revised repeatedly, so conversion is repeatable:

```
your.urdf  +  <name>.conversion.yaml  →  <name>.xml  +  decimated meshes
                  ↑ hand-maintained; everything URDF cannot express lives here
```

The sidecar config holds: rotor / imu / ee site positions and orientations, the
per-rotor `c_t` / `km` / `spin` / `ω_min` / `ω_max` arrays of §2.5, actuator gains
and limits, collision primitive substitutions, `<option>` overrides, mesh
scaling, and the world (floor, light, ENU markers — a URDF describes one robot
and has no syntax for a scene). Re-exporting from CAD then only means re-running
the script, and no hand edits are lost.

Angles accept radians, `{deg: 45}`, or `"45deg"`; unknown keys are rejected
rather than ignored, since a dropped `omega_max` silently leaves the placeholder
in place and still flies.

The script uses MuJoCo 3.13's `MjSpec` to inject programmatically, then self-checks
and prints:

- `freejoint` exists and is `base_link`'s first joint
- `rotor*` sites are on `base_link`, contiguously numbered, count matching `spin`
- `imu_accel` / `imu_gyro` exist and are 3-axis
- CAD meshes survived, and the vehicle has colliding geometry and a floor
- arm joints are limited, with non-zero `damping` / `armature`
- total mass / CoM / inertia matrix, to cross-check against SolidWorks

`--check-only` runs all of it without writing. A failed check is a non-zero exit
and no MJCF, so a bad conversion cannot be flown by accident.

FLU orientation cannot be checked automatically — that rests on §3.1 and a manual
review.

### Three things the conversion has to fight MuJoCo about

All three produce a model that loads, has the right mass, and flies — wrongly.

**`discardvisual` defaults to true on the URDF path, and applies during
parsing.** It deletes every geom that does not collide, which is exactly what the
CAD meshes become once §3.2's rule is applied. Clearing the flag on the loaded
spec is too late — the geometry is already gone — so the script injects
`<mujoco><compiler discardvisual="false"/></mujoco>` into the URDF text before
parsing. A link whose only geometry is `<visual>`, like every arm link here, is
what gets lost; nothing in a headless run reveals it.

**SolidWorks writes `lower=upper=0` when limits were never set in CAD, and MuJoCo
compiles that as *unlimited*.** So the export's own default is not a locked joint
but a free one, carrying a position actuator whose `ctrlrange` spans nothing.
Zero-length limits are a hard error in the sidecar, and
`test_that_zero_limits_really_do_compile_to_unlimited` pins the premise against
MuJoCo itself.

**MuJoCo's binary STL decoder refuses any file above 200 000 faces.** A raw
SolidWorks export routinely exceeds it — `base_link` here is 303 527 faces — so
decimation is a requirement, not a performance preference, and the error without
it names the STL rather than the cause. The script decimates by vertex
clustering: no extra dependency, and the error is bounded by one grid cell
(2 mm on a 0.95 m body at the default budget). These meshes are visual only, so
shape fidelity matters and watertightness does not.

---

## 7. Data still needed

The first CAD export has landed. What it answers and what it does not is in
`AGENTS.md` §4 — geometry and mass properties came through; joint limits and motor
numbers did not. Figures stay with the filled sidecar in the private repo.

Still needed from CAD:

```
rotor i spin               = CCW / CW   ×8   confirm against §4's order and ESC wiring
per-joint range            = (lo, hi) rad    THE blocker: exported as 0/0
per-joint zero and sign convention           arm_cmd is an absolute angle
per-joint gear ratio + stall torque → forcerange and armature
rotor radius / blade count / I_prop
end-effector reference point = optional, see §8
```

Hub positions, deck spacing and prop radius were **measured off the exported
mesh** rather than read from CAD reference frames — the export carried none for
the rotors. They are in the sidecar and the script cross-checks mass and CoM
against the CAD figures, but the hub coordinates themselves are worth confirming
in SolidWorks.

**Motor and propeller data has landed** and §2.5's coefficients are fitted from it
rather than assumed:

```
c_t per rotor              [N/(rad/s)²]  fitted, least squares through the origin
km  per rotor              [m]           fitted from the torque column
ω_max per rotor            [rad/s]  at flight battery voltage, from the sheet's
                                    full-throttle RPM — not the nominal KV
                                    product, which is 25 % high
ω_idle                     [rad/s]  STILL A CHOICE: datasheets start near 0.40
                                    throttle, so armed idle is not in them.
                                    Bench-measure it rather than extrapolating
                                    the curve to zero, which overestimates badly
lower-deck c_t discount    = still not modelled   (§5 suggests 0.75–0.85)
```

`c_t` and `ω_max` must come from the same sheet: thrust is `c_t · ω²`, so only the
product is physical, and a measured `c_t` against the placeholder `ω_max` silently
rescales every thrust in the model — it raises nothing and still flies. The
conversion script rejects that combination outright.

A fitted `km` is rarely PX4's `CA_ROTOR*_KM` default of 0.05, and the airframe file
must carry the fitted value; left at the default the allocator expects the wrong
yaw authority, which presents as sluggish yaw rather than as an error.

`ω_idle / ω_max` and `thrust_to_weight` set `MPC_THR_HOVER` and nothing else does,
so a platform with its own motor numbers needs its own value — the quad's does not
carry over. `MujocoPhysics` logs the hover command at load, and the conversion
script prints it, so a mismatch is visible before flight.

`vehicle.py`'s `OMEGA_MAX_PLACEHOLDER` / `OMEGA_IDLE_PLACEHOLDER` /
`CT_PLACEHOLDER` remain the fallback for any model that supplies none of this, and
still warn on every load (§2.4). A new platform whose motors are not yet chosen can
therefore be modelled and flown before they are, with the real numbers landing
later and reworking nothing.

---

## 8. Open questions

**The lower-deck `c_t` discount.** §2.5 provides the per-rotor array and the sidecar
now carries per-rotor `c_t`, so the mechanism is in place and this is only a number
to pick. Undecided, and currently **not applied**: the datasheet is for an isolated
rotor, so all 8 carry the same value and the lower deck is modelled too strong.

**Whether end-effector pose joins the side channel.** An `ee` site plus end-effector
pose in `ground_truth` makes visual servoing or impedance control on the ROS 2 side
much easier. The site itself now exists in the model (the sidecar places it), so
only the schema side is open — and later it costs a version bump
(`SCHEMA_VERSION`, [sidechannel.py:28](src/mujoco_px4_sitl/sidechannel.py#L28)).

**How the sidecar's rotor parameters reach `RotorModel`.** The sidecar holds
measured `c_t` / `km` / `ω_max`, the script validates and prints them, and nothing
consumes them — `main.py` builds no `RotorModel`, so an 8-rotor model cannot load
(`AGENTS.md` §2). Two candidate mechanisms, undecided: a `--rotors` path on the CLI,
or having the conversion script emit them into the MJCF as `<custom><numeric>` so
the model is self-contained. The second keeps `src/` free of a YAML dependency and
cannot drift from the model it describes.

**Whether the arm needs a control rate distinct from the physics rate.** Listed as
open in Phase 7 of `IMPLEMENTATION_PLAN.md`.

---

## 9. Current state and next steps

See `AGENTS.md`. It owns the state of the work, the list of what is unimplemented,
and the ordered next steps — several of which are the code changes §2.5 calls for.
Keeping them in one place avoids two task lists drifting apart.

