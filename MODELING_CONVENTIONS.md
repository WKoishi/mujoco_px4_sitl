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
- Thrust model `T(u) = k_thrust · u²`, first-order motor lag
  `time_constant = 0.02 s`. **Being replaced by an ω-based parametrization — see
  §2.5.**
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

**`spin` is a hardcoded 4-tuple**
([vehicle.py:45](src/mujoco_px4_sitl/vehicle.py#L45)), and `build_physics` never
passes a custom `RotorModel` ([sim.py:238](src/mujoco_px4_sitl/sim.py#L238)). An
octorotor raises `ValueError` at
[vehicle.py:84-88](src/mujoco_px4_sitl/vehicle.py#L84-L88) — a hard error, not a
silent one. **X8 requires fixing this first.**

### 2.3 The arm: commands never reach the physics

The side channel parses `arm_cmd` into `ArmCommand(mode, values)`
([sidechannel.py:95-103](src/mujoco_px4_sitl/sidechannel.py#L95-L103)), but
**nothing in `src/` writes `data.ctrl`** — the command is received and dropped.
Phase 7 work.

### 2.4 The side effect of auto-calibrated `k_thrust`

By default it is calibrated from `body_subtreemass` so that hover sits at command
0.5 ([vehicle.py:100-109](src/mujoco_px4_sitl/vehicle.py#L100-L109)), fixing
thrust-to-weight at 4.0. Upside: any arm mass still flies. Downside: **the motors
quietly get stronger**, hiding a real platform that would be underpowered.

Under §2.5 this becomes an opt-in fallback rather than the default, and it should
warn when the resulting thrust-to-weight is implausible. Dropping it outright would
turn "heavy arm" into "will not take off", which is correct physics but presents as
a broken model.

### 2.5 Planned: ω as an explicit state

The current model folds `C_T` and `ω_max²` into the single scalar `k_thrust`, which
cannot be taken apart again. **Rotor speed is what every aerodynamic effect in §5
needs** — advance ratio, flapping, coaxial interference and gyroscopic precession
all take ω, not a normalized command. So `RotorModel` moves to:

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

**The one real knock-on cost.** A non-zero idle speed moves the hover operating
point. For an X8 at ~3 kg with `ω_max = 1100 rad/s`, holding full-command
thrust-to-weight at 4.0:

| | Current | ω-based |
|---|---|---|
| Hover command | 0.500 | **0.450** |
| Normalized local gain `d(T/W)/du` | 4.000 | **3.636** (0.909×) |
| Thrust at zero command | 0 | 3.3 % of weight |

So `MPC_THR_HOVER` has to be recomputed, and the argument in
[px4/22001_mujoco_quad:60-63](px4/22001_mujoco_quad#L60-L63) and
[vehicle.py:34-37](src/mujoco_px4_sitl/vehicle.py#L34-L37) — that the quadratic
plant and PX4's linear allocator share a slope at `u = 0.5`, so `THR_MDL_FAC` can
stay 0 — has to be redone. Not a blocker, but it is the only genuine knock-on
change, and missing it presents as altitude oscillation in hover.

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

The current rotor model is pure feedforward — thrust is a function of command only,
and `vehicle.py` **reads no velocity state at all**. None of the effects below need
separate links, but all of them need rotor speed, which is why §2.5 promotes ω to a
state:

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

The CAD will be revised repeatedly, so conversion has to be repeatable:

```
your.urdf  +  models/<name>.conversion.yaml  →  models/<name>.xml
                  ↑ hand-maintained; everything URDF cannot express lives here
```

The sidecar config holds: rotor / imu / ee site positions and orientations, the
per-rotor `c_t` / `km` / `spin` / `ω_min` / `ω_max` arrays of §2.5, actuator gains
and limits, collision primitive substitutions, `<option>` overrides, and mesh
scaling. Re-exporting from CAD then only means re-running the script, and no hand
edits are lost.

The script uses MuJoCo 3.13's `MjSpec` to inject programmatically, then self-checks
and prints:

- `freejoint` exists and is `base_link`'s first joint
- `rotor*` sites are on `base_link`, contiguously numbered, count matching `spin`
- `imu_accel` / `imu_gyro` exist and are 3-axis
- total mass / CoM / inertia matrix, to cross-check against SolidWorks

FLU orientation cannot be checked automatically — that rests on §3.1 and a manual
review.

---

## 7. Data still needed

From CAD:

```
rotor i hub position (FLU) = (x, y, z)  ×8   4 per deck, z as built
rotor i spin               = CCW / CW   ×8   copy §4's order
rotor radius / blade count / I_prop
arm mount point on the base = arm_joint0's origin
per-joint range            = (lo, hi) rad
per-joint gear ratio + stall torque → forcerange and armature
end-effector reference point = optional, see §8
```

From motor and propeller data, for §2.5:

```
c_t per rotor              [N/(rad/s)²]
ω_max per rotor            [rad/s]  at flight battery voltage
ω_idle                     [rad/s]  armed-but-idle
lower-deck c_t discount    = ? or "not modelled"   (§5 suggests 0.75–0.85)
```

If the motors are not chosen yet, say so — §2.4's auto-calibration stays available
as a fallback, and the platform-specific numbers can land later without reworking
the model.

---

## 8. Open questions

**The lower-deck `c_t` discount.** §2.5 provides the per-rotor array, so this is now
a number to pick, not a design question. Undecided.

**Whether end-effector pose joins the side channel.** An `ee` site plus end-effector
pose in `ground_truth` makes visual servoing or impedance control on the ROS 2 side
much easier. Cheap now; later it costs a schema version bump
(`SCHEMA_VERSION`, [sidechannel.py:28](src/mujoco_px4_sitl/sidechannel.py#L28)).

**Whether the arm needs a control rate distinct from the physics rate.** Listed as
open in Phase 7 of `IMPLEMENTATION_PLAN.md`.

---

## 9. Current state and next steps

See `AGENTS.md`. It owns the state of the work, the list of what is unimplemented,
and the ordered next steps — several of which are the code changes §2.5 calls for.
Keeping them in one place avoids two task lists drifting apart.

