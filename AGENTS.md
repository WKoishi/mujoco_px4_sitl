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

**The X8 has flown the Phase 5 profile**, arm held at zero, and passes it with
the quad's numbers: hover 0.117 m horizontal / 0.202 m vertical max against
ground truth, corners 0.01–0.05 m, `brake=0 timeouts=0`, no failsafe.
`IMPLEMENTATION_PLAN.md` phase 5 has the table. PX4's own `hover_thrust_estimate`
settled on 0.2162, exactly the generated `MPC_THR_HOVER`, so the sidecar's thrust
and the airframe agree through PX4's loop and not only on paper.

Of the three dynamic unknowns that flight was meant to settle:

- **The index→ESC mapping is consistent with the model.** Yaw steps move the
  commanded way, roll/pitch stay under 2° through them, and hover yaw error is
  0.2°. A mismatch would have diverged or cross-coupled. Still unconfirmed
  against the *wiring*, which no simulation can settle.
- **Coaxial yaw authority is weak, and PX4's default gains do not suit it.**
  Measured, and the cause is known — see below.
- **The arm's mass moving in flight** is now flown; see below.

**PX4's rate gains act on a normalized torque**: the allocator scales each axis
so that ±1 means the vehicle's own authority on that axis. So the physical loop
gain is the default gain times (authority / inertia), and on this X8 that ratio
is about 20× below the quad's in yaw, 13× in pitch and 5× in roll. With stock
`MC_*RATE_*` the yaw-rate loop crosses over *below* the yaw attitude loop
(`MC_YAW_P 2.8`), and a 20° yaw step overshoots 50–70 % and takes 5–8 s to
settle. The quad does the same step in 0.6 s with 1–4° overshoot. Pitch is
marginal (16 % overshoot on a 5° step), roll is fine (≤ 1.4 %). Yaw also runs
out of authority at hover: on a 90° step a motor sits at zero and yaw
acceleration tops out near 80°/s², so a 200°/s rate limit cannot be braked in
time.

**The generated airframe now carries derived gains**, and the X8 has flown them.
`--emit-airframe` computes each axis's plant gain from the model and the
sidecar. It raises `MC_*RATE_K` only until the rate loop is 4× faster than the
attitude loop (capped at PX4's documented 5), and sets `MC_*RATE_MAX` to the
acceleration left at hover over the attitude P. The airframe template says
why. For this X8 that is roll stock, pitch 2.18, yaw 5 (the rule asks for 9),
and yaw limited to 29.6°/s. With those gains a 20° yaw step settles in 0.7 s
with at most 2° overshoot, a 90° step overshoots at most 0.6°, pitch overshoots
2 %, and the Phase 5 profile still passes. `IMPLEMENTATION_PLAN.md` phase 5 has
the tables.

**The first version of the rule was wrong, and flight caught it.** It matched
the quad's physical loop gain, which put roll at K 4.8. Hover and the
regression profile were fine, but after the first yaw step drove a motor to
zero, roll went into a limit cycle that lasted over a minute. At the same loop
gain this X8 has about 4.5× less acceleration headroom than the quad, so it
saturates on much smaller signals. The rule now asks only for enough speed.
This matters for the arm: shoulder pitch and base yaw are exactly what the arm
disturbs.

Motors are chosen: `c_t`, `km` and `ω_max` are fitted from the manufacturer's
data and live in the private sidecar (§4). `ω_idle` is not in that data and is
still a choice, and `vehicle.py`'s placeholders remain the fallback for any model
that does not supply its own — which is now only the quad.

**The arm is driven** through [arm.py](src/mujoco_px4_sitl/arm.py);
`README.md` ("Driving the arm") has the interface. Two decisions came with it:

- **A command watchdog on simulated time**, on by default, whose timeout action
  (`freeze`, `keep`, `limp`) is configurable: it models the arm driver, which a
  lockstep stall must not trip, and what the arm does when its controller
  stalls is what the private research studies. A command may carry
  `state_time`, making its age the age of the data behind it.
- **Propeller intrusion is checked every frame on the reached pose, never
  blocked**, against discs drawn from a per-rotor `radius` (§4).

Flown the same day (`IMPLEMENTATION_PLAN.md` phase 7): PX4 held position
through every arm move within 2° of attitude, **except after a watchdog stall,
when the first fresh command landed as a step and cost 7.8° of pitch**. Servo
droop moved reached poses across a disc in both directions, so **the commanded
pose does not predict an intrusion**.

**The research controller runs in process** ([control.py](src/mujoco_px4_sitl/control.py),
decided in §3), and has flown the X8. It is called from the lockstep loop at
chosen sample instants of simulated time; its arm command lands after a chosen
delay, less a seeded drop schedule, so the command age has a known bound. It
reads the arm's encoders and PX4's EKF2 estimate over MAVLink
([px4link.py](src/mujoco_px4_sitl/px4link.py)), the path a companion computer
uses. Ground truth goes only to a recorder. `run_sitl.sh --px4-only` starts PX4
for it. Two flights under one schedule (`IMPLEMENTATION_PLAN.md` phase 7) gave
identical sample instants and drops and a bit-identical arm trajectory until
arming, while the median estimate age went from 8 ms to 4 ms. **The arm leg is
chosen; the PX4 legs are only measured.**

**Making the PX4 legs chosen is the next step, and it is decided but not built.**
Phase 2 of the topology was probed and decided on 2026-09-26: strict lockstep
as the only regime, IMU → actuator exactly one frame, estimates fetched at a
chosen age of at least one frame, setpoints behind a barrier (§3 has the
decisions and the work list). Until it lands, everything above about the loop
and the status line describes today's code.

---

## 2. What is not implemented

| Gap | Where | Consequence |
|---|---|---|
| **`--hold-pose` makes the arm weightless** | `_pin_pose` restores the base only between frames, so within one the vehicle free-falls | no droop, and a `limp` arm does not fall. Phase 3's IMU table is unaffected; testing the arm on a pinned vehicle is not. `tests/test_arm.py` welds the base instead. The fix is a support wrench at the subtree CoM |
| Propeller clearance is idealized | `PropellerMonitor` in [arm.py](src/mujoco_px4_sitl/arm.py) | flat discs (the real sweep is about ±7 mm thick at the root), capsules and spheres only, once per IMU frame, against collision primitives fitted by eye (§4). The conversion's envelope survey is kinematic, so droop can move its poses across a disc either way. Margin is the consumer's to add |
| Arm joint zero and sign convention | unverified against hardware | `arm_cmd.values[i]` is an absolute angle, so a flipped sign drives the real arm the wrong way. The ranges themselves are settled |
| Arm joint state reaches only an in-process controller | `ground_truth.arm` carries command age, staleness and propeller clearance, not `qpos` | an out-of-process consumer (a ROS 2 bridge, a plot) sees no joints. No longer a controller problem (§3); add it to the side channel when a monitoring consumer needs it |
| Arm encoders are ideal | `JointReading` in [arm.py](src/mujoco_px4_sitl/arm.py) is MuJoCo's joint state at the sample instant | no quantisation, noise or latency on the controller's joint feedback. Model them from the servo's datasheet once it is chosen |
| Arm has position servos only | the generated MJCF has one position actuator per joint (`arm_act*`); `arm.py` refuses any other kind, and rejects `mode: "torque"` | a torque or velocity joint interface cannot be simulated yet. MuJoCo fixes the actuator type at compile time, so either the conversion script emits more actuators or the writer applies `qfrc_applied`. The servo gains are provisional too, and decide intrusions as well as tracking: their gravity droop moved reached poses across a disc (§1) |
| Only the arm leg's delay is chosen | `control.py` schedules sample, delay and drops on simulated time. Which EKF2 estimate has arrived by a sample, PX4's IMU → actuator response, and setpoints the controller sends PX4 are wall-clock (§3) | a data-age argument can be tested at a chosen age on the arm. On the base, `estimate_age` and `px4_lag` measure the age exactly but cannot set it, so runs differ (median estimate age 8 ms, then 4 ms, under one schedule). Decided, not implemented: phase 2 in §3 |
| Two runs of one schedule drift apart | PX4's `sensor_*_sim` draw their noise from one shared `rand()` whose position boot decides (`IMPLEMENTATION_PLAN.md` §3.3), and within a frame PX4's threads race (§3.2 there) | identical until arming, then up to about a metre apart in hover, under any loop, so a single pair cannot resolve an effect smaller than that; compare over repeated runs. Seeded simulator-side sensors (strategy B, deferred, §3) brought two strict runs to 2–4 cm. Bit-identical runs would need PX4 changes |
| Attitude gains are sized for the arm at home | `attitude_gains` uses the inertia at the model's home pose | an arm that moves in flight changes the inertia and the gains do not follow. Rigid-arm inertia is what the Phase 7 disturbance test starts from, so it is the right starting point, not the whole answer |
| IMU has no noise or bias | `sim.py` sends MuJoCo's ideal accel/gyro, and PX4's `simulator_mavlink` only quantizes them (`SimulatorMavlink.cpp:198-270`): under MAVLink HIL the simulator owns IMU noise. Baro, mag and GPS do get noise, from PX4's `sensor_*_sim` | EKF2's bias estimation is never exercised, and estimate errors look better than they will be -- including the estimate an in-process controller is handed. The derived attitude gains were verified on a noiseless gyro, and noise is what the rate D term suffers from, so re-fly the step tests once it lands. Add white noise and a bias random walk on the `HIL_SENSOR` path only, sized from the chosen IMU's datasheet: `SimState.gyro_frd` also feeds ground truth (`HIL_STATE_QUATERNION`, the side channel), which must stay clean. If strategy B lands first, its sensor module is the place (`IMPLEMENTATION_PLAN.md` §3.3) |
| No aerodynamics | ω is available but nothing reads it | the effects `MODELING_CONVENTIONS.md` §5 lists are expressible, none are expressed |
| Coaxial `c_t` discount | all 8 rotors carry the isolated-rotor value | lower deck is modelled 15–25 % too strong (`MODELING_CONVENTIONS.md` §5 suggests 0.75–0.85). Deliberately deferred so the first X8 flight introduces one unverified quantity, not two: it moves `MPC_THR_HOVER` to 0.2377 / 0.2456 / 0.2541 at 0.85 / 0.80 / 0.75, so the sidecar's `c_t` and the airframe must change together |
| No uXRCE-DDS agent or `px4_msgs` on this machine | PX4's side is ready: `px4_sitl_default` starts `uxrce_dds_client` on UDP 8888, subscribed to `/fmu/in/vehicle_thrust_setpoint` and `/fmu/in/vehicle_torque_setpoint`, and nothing on the host answers | the normalized thrust/torque interface has no MAVLink offboard path in v1.17, so it cannot be flown until an agent and `px4_msgs` are installed. Attitude and body-rate setpoints work over MAVLink today. Environment rather than code: `src/` stays free of ROS imports (§6). DDS setpoints cannot use phase 2's MAVLink barrier (`IMPLEMENTATION_PLAN.md` §9) |

`scripts/manipulability.py` answers the kinematic half of the arm question —
whether a pose is reachable at all, per voxel, with any subset of joints frozen.
It needs no PX4 and no physics step, and says nothing about flight: rank is not
thrust.

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
model whose rotor count still matches flies with the wrong thrust.

Flying the X8 is **done** (§1): it passes the Phase 5 profile. So are its
attitude gains: `--emit-airframe` derives them from the model and the sidecar
the same way it derives `MPC_THR_HOVER`, so they follow the arm and the motors
when either changes (§1). A private research controller that replaces PX4's
rate loop needs none of it.

Driving the arm is **done** (§1). A future writer that applies forces to the
arm owns its own clearing: `clear()` zeros only `base_link`'s `xfrc_applied`.
The research-controller topology is **decided**; its first phase is built and
its second decided (below). What remains:

1. **Phase 2 of the topology: strict lockstep and the PX4 legs.** Decided
   2026-09-26, not implemented. The work list is below; this is the next step.
2. **The coaxial `c_t` discount**, as its own step with its own flight, after
   phase 2 so the two stay separable. It moves the plant gain as well as the hover
   point, so regenerate the airframe and fly the attitude steps again.

**Strategy B — baro, mag and GPS synthesized by the simulator, seeded — waits
until a use needs it,** such as single-pair counterfactuals, sensor noise as a
controlled variable, or the IMU noise gap (§2). `IMPLEMENTATION_PLAN.md` §3.3 has
the evidence and what it takes. Until then two runs of one schedule drift apart
by up to a metre (§2).

Aerodynamic effects are deliberately **not** on this list. They come after Phase 7's
exit criterion; `RotorState.omega` is what makes them expressible.

### The research-controller boundary

**Decided 2026-09-25: in process.** The research controller is called
synchronously from the lockstep loop (`control.py`), and `sim/run_x8.py` in the
private repo starts PX4 with `run_sitl.sh --px4-only` and the simulator with
`run(cfg, controller=...)`. The controller takes the base state from PX4's EKF2
over MAVLink, as on the vehicle, not from MuJoCo. The requirements behind it
are the research's: data-age bounds tested at a *chosen* age, counterfactual
comparisons under identical timing, stalls injected by design, and batches
faster than real time. Out of process, all four could only be measured.

**What the earlier out-of-process reasoning got wrong.** It was chosen for
isolation, "a slow solver cannot stall lockstep". But under lockstep a stall is
harmless to PX4: its clock only advances with our `HIL_SENSOR`, and
`SimulatorMavlink`'s 1 s wall-clock `poll` timeout only logs. The real costs of
a synchronous controller are the sim/wall ratio and the pacer's catch-up burst
after a slow call, which widens PX4's lead. The loop now removes the
controller's time from its pacing, so only the ratio remains.

**In process chose only the arm leg.** In today's loop three legs are still
wall-clock: which of PX4's outputs a frame uses (the brake lets the loop run
ahead), which EKF2 estimate a sample sees, and when a setpoint the controller
sends reaches PX4's uORB. Phase 2 makes all three chosen.

### Phase 2: decided 2026-09-26, not implemented

Probed before designing. The PX4 facts are in `IMPLEMENTATION_PLAN.md` §3.2
("What an answer proves") and §3.9, the measurements in phase 7 ("The PX4
legs"), and the decided design at the end of §3.2 and of §3.9. The decisions:

- **Strict lockstep is the only regime.** Once PX4 has answered, every frame
  waits for the answer stamped with its time. Today's loop stays only as the
  fallback before PX4's first answer and for a frame whose answer times out.
- **IMU → actuator is exactly one frame.**
- **Estimates are fetched after each answer, at a chosen age of at least one
  frame.** No `/proc`, no age 0.
- **Setpoints and commands go behind a PING barrier, at a chosen delay of at
  least one frame.**
- **Strategy B later, when needed** (above).

What that buys, per leg: IMU → actuator chosen and proven by the stamp;
estimates chosen, aged `d` or `d` + 1 frame by EKF2's 8 ms cadence, completeness
checked after the fact; body-rate setpoints chosen exactly, attitude setpoints
chosen plus PX4's own 0–1 frame, a race inside PX4 that can be measured but not
chosen. Two runs share their timing, not their state (§2). Probed speed, unpaced:
quad about 12×, X8 about 8×.

Work list, in order:

1. **`loop.py`: the strict regime.** The frame `[t_k, t_{k+1})` is driven by the
   answer stamped `t_{k−1}`, and `HIL_SENSOR(t_{k+1})` is not sent before the
   answer stamped `t_k` or its wall-clock timeout. Today's pacer and brake, with
   the invariant of plan §3.2, run until PX4's first answer and for timed-out
   frames, which are counted as unproven. The pacer stays as a ceiling; add an
   unpaced setting (`Config.validate` rejects `speed_factor <= 0` today). On the
   status line, answered and unproven counts replace `px4_lag`.
   `tests/test_loop.py`, against fake PX4s: one answering every frame with the
   right stamp gives zero brakes, zero timeouts and each answer applied exactly
   one frame later; one silent through boot does not deadlock; one falling
   silent mid-run gives unproven frames, never a fatal error; an answer with
   another frame's stamp is not taken as this frame's.
2. **`px4link.py`: the host's API link.** Send with a PING barrier, whose
   wall-clock timeout is loud and recorded. After every answer, not only at
   samples, `REQUEST_MESSAGE(ODOMETRY)` behind a barrier: the stream sends only
   the newest output, and the one a sample needs is older. Before the first
   barrier, silence the periodic `ODOMETRY` with an interval of 2·10⁹ µs, not
   −1, and wait for its `COMMAND_ACK` while frames advance without barriers.
   Refuse on the barrier path anything that ends in
   `configure_stream_threadsafe()` (plan §3.9). Keep the origin handling. Tests:
   a fake peer that echoes PING in order, and one that stalls the way PX4's
   blocked receive thread does.
3. **`control.py`: the schedule and the controller's outputs.** `Schedule` gains
   an estimate delay and a setpoint delay, whole IMU frames, at least one. The
   observation's estimate is the newest output with sample time ≤ `t_k − d`,
   with an age bound; a sample whose estimate turns out late is flagged in the
   record. A setpoint returned at `t_k` is in uORB before PX4 processes the
   frame at `t_k + d_sp` (plan §3.9). `Controller.step` may return PX4 setpoints
   and commands besides the arm targets; the shape is the implementer's, and
   `src/` stays free of ROS. Arming and takeoff at simulated times go the same
   way, which replaces the wall-clocked GCS script for runs meant to be
   compared.
4. **Documents.** Plan §3.2: the decided regime becomes the description, today's
   loop the fallback; plan §3.9's decision paragraph likewise; plan §7 for the
   new status line. `README.md`'s health check and in-process controller
   sections. This file. `sim/run_x8.py` in the private repo says the PX4 legs
   are wall-clock and exposes only the arm schedule; update it with the user,
   who stages work of their own in that repo.
5. **Flights**, each recorded at the phase it belongs to: the Phase 5 profile on
   the quad and the X8 against their baselines; the X8's yaw and pitch steps,
   since IMU → actuator moves from mostly one frame to exactly one; the Phase 7
   in-process controller twice under one schedule, armed at a simulated time,
   where sample instants, drops and estimate ages should now all match.
6. **Acceptance: repeat the probe's measurements through the real loop.** Every
   frame answered after PX4's first answer; a body-rate setpoint used by its
   intended frame on every frame, idle and with every core busy; no late
   estimate at an age of one frame, idle and loaded; speed of the order above.

The probe that produced the numbers lives in `../phase2_probes/`, next to the
PX4 checkout: local, not versioned, with a README. It has a working strict loop,
the barrier, the estimate fetch, the thrust-PRBS lag measurement, and seeded
simulator-side sensors for strategy B. Use it as a reference; whether its checks
become a script in this repo is the implementer's call. The plan carries all the
facts without it.

Traps the probes hit:

- **The receive thread blocks on simulated time behind `SET_MESSAGE_INTERVAL`**
  (plan §3.9), so a barrier sent after one hangs until its timeout.
- **An answer does not prove the frame's estimate** (plan §3.2). It looks as if it
  does on an idle machine; that is why the age is at least one frame.
- **The barrier looks unnecessary when idle.** Without it, under load, two thirds
  of the setpoints missed their frame.
- **`SET_ATTITUDE_TARGET` publishes its setpoint only in Offboard**; before the
  switch only `offboard_control_mode` goes through. Semantics, not timing.
- **PX4 starts answering 4–53 frames after it connects**, so the fallback is
  needed however early strict mode would like to start.
- **A probe that starts PX4 itself meets the exit trap** in
  `IMPLEMENTATION_PLAN.md` §7.

Leave room for uXRCE-DDS thrust/torque; how to time its setpoints is open
(plan §9).

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

- **Arm joint limits: every one is `lower=upper=0`.** Worse than it looks in the
  export — MuJoCo reads 0/0 as *unlimited*, so the export's default is a free
  joint, not a locked one. **The sidecar's ranges are not a placeholder for this:
  they are the mechanical limits, read off each joint's structure rather than out
  of the export.** What the export withheld was the *encoding*, and the sidecar
  supplies it. Still unverified is the zero-position and sign convention, since
  `arm_cmd.values[i]` is an absolute angle.
- **Spin directions and index→ESC mapping.** The sidecar assumes
  `MODELING_CONVENTIONS.md` §4's PX4 order. Flight shows it consistent with the
  model (§1); it still needs confirming against the wiring, since a mismatch
  there presents as yaw drift on the real vehicle.

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

**The arm's envelope and the propeller discs intersect, inside the mechanical
limits.** Not a near miss: a small fraction of the poses the mechanism genuinely
permits put an arm collision capsule *inside* a disc, measured on the model with
the real ranges (the figures are in the private sidecar). So the limits cannot be
the guard — they are the mechanism's, and the mechanism can do this. Nothing in
the model stops a commanded angle from doing it either. Only the front rotors
are reachable, on both decks, and only by the last link.

Decided 2026-09-25: in simulation it is **detected and reported, never
blocked** (§2). Check the pose the arm actually reaches, its collision capsules
against the disc volumes each step, rather than the commanded one, since the
servo lags its command. Blocking is a real-vehicle safety layer and belongs in
the arm driver or the companion computer, outside this repo. The joint limits
cannot do it on either side. Implemented the same day (§1).

**Measuring the discs corrected two inputs.** Each rotor's `radius` is the blade
tip off the base mesh; the earlier hand survey had used a smaller radius of no
recorded origin. And the rotor sites now sit on the blade planes, which the
upper deck's did not — harmless while z carried no torque and no `PZ`, but z
places the disc. The airframe regenerated byte-identical. The conversion's
self-check now reruns the envelope survey with the simulator's own check, and
on the old geometry it reproduces the hand survey exactly.

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
  will not tell you otherwise. `IMPLEMENTATION_PLAN.md` §7 diagnoses both. Once
  phase 2 lands (§3), the brake acts only before PX4's first answer, and the
  count to watch in flight is unproven frames.
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
python sim/run_x8.py            # PX4 via run_sitl.sh --px4-only; simulator and
                                # research controller in this process
```

Without a controller, driving the arm over the side channel instead:

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
