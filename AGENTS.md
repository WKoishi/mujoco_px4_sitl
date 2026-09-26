# Handoff

Start here when picking up this work: current state, what is not built, what to
do next, and how to work without drowning the session (§6). No interface contract,
no model authoring rules, no history: §5 says which document owns what, and git
has the reasoning of each step. §7 has the launch lines.

---

## 1. Where things stand

Phases 0–7 are met, and `python -m pytest` is green with no PX4 build and no GL.
`IMPLEMENTATION_PLAN.md` holds every measured baseline at the phase that produced
it. **Those are the regression reference: do not copy them here, and do not delete
them when the platform changes.**

- **Quad** (`models/quad_x.xml`) flies the Phase 5 profile inside the GPS-noise band.
- **X8 + 5-DoF arm** is generated from the CAD export by `scripts/urdf_to_mjcf.py`
  into the private repo, together with its PX4 airframe (`22002_mujoco_x8`). The
  airframe carries the measured allocation, `MPC_THR_HOVER` and attitude gains
  derived from the model (`px4/mujoco_x8.airframe.template` says how). It flies
  Phase 5 and its attitude steps (plan phase 5).
- **The arm** is driven through [arm.py](src/mujoco_px4_sitl/arm.py): position
  servos, a command watchdog on simulated time, propeller intrusion reported on
  the reached pose and never blocked (`README.md`, "Driving the arm").
- **The research controller runs in process** ([control.py](src/mujoco_px4_sitl/control.py)),
  on simulated time, reading the arm's encoders and PX4's EKF2 estimate over
  MAVLink ([px4link.py](src/mujoco_px4_sitl/px4link.py)) as a companion computer
  would, and sending PX4 setpoints and commands the same way. Ground truth goes
  only to a recorder.
- **Every leg to the vehicle has a chosen delay** since phase 2 of the topology
  (2026-09-26): strict lockstep with IMU → actuator exactly one frame, estimates
  at a chosen age, setpoints and commands behind a PING barrier before a chosen
  frame (plan §3.2, §3.9). Acceptance passed on the quad and the X8, idle and
  loaded; `scripts/fly_in_process.py acceptance` repeats it in under a minute.

---

## 2. What is not implemented

| Gap | Where | Consequence |
|---|---|---|
| **`--hold-pose` makes the arm weightless** | `_pin_pose` restores the base only between frames | no droop, and a `limp` arm does not fall; `tests/test_arm.py` welds the base instead. Fix: a support wrench at the subtree CoM |
| Propeller clearance is idealized | `PropellerMonitor` in [arm.py](src/mujoco_px4_sitl/arm.py) | flat discs (the real sweep is about ±7 mm thick at the root), capsules and spheres only, once per frame, primitives fitted by eye. Margin is the consumer's to add |
| Arm joint zero and sign convention | unverified against hardware | `arm_cmd.values[i]` is an absolute angle; a flipped sign drives the real arm the wrong way |
| Arm joint state reaches only an in-process controller | `ground_truth.arm` has no `qpos` | an out-of-process consumer sees no joints; add it to the side channel when a monitoring consumer needs it |
| Arm encoders are ideal | `JointReading` | no quantisation, noise or latency; model them once the servo is chosen |
| Arm has position servos only | one position actuator per joint; `arm.py` refuses others and `mode: "torque"` | no torque or velocity interface. The servo gains are provisional, and their droop decides intrusions as well as tracking |
| Two things in PX4's legs are measured, not chosen | EKF2's cadence phase, set at boot; PX4's attitude/rate thread race (plan §3.2, §3.9) | the estimate's age is `estimate_delay` or one frame more, and which samples get the shorter one can swap between runs. An attitude setpoint is one frame later than a body-rate one on most frames, the same frame on 0–98 %, by load. Accepted; `EKF2_PREDICT_US 4000` would remove the first, not tried |
| Two runs of one schedule drift apart | PX4's `sensor_*_sim` share one `rand()` whose position boot decides (plan §3.3) | identical until arming, then up to about a metre apart (0.34 m for the strict X8 pair). Compare over repeated runs. Seeded simulator-side sensors (strategy B, §3) brought a pair to 2–4 cm |
| Attitude gains are sized for the arm at home | `attitude_gains` uses the home-pose inertia | the gains do not follow an arm that moves in flight |
| IMU has no noise or bias | `sim.py` sends ideal accel/gyro; under MAVLink HIL the simulator owns IMU noise | EKF2's bias estimation is never exercised, and estimates look better than they will be. Add noise on the `HIL_SENSOR` path only (ground truth must stay clean), sized from the IMU's datasheet, then re-fly the step tests |
| No aerodynamics | ω is available, nothing reads it | `MODELING_CONVENTIONS.md` §5 lists the effects |
| Coaxial `c_t` discount | all 8 rotors carry the isolated-rotor value | lower deck 15–25 % too strong. Moves `MPC_THR_HOVER` to 0.2377 / 0.2456 / 0.2541 at 0.85 / 0.80 / 0.75: sidecar and airframe change together. Next step (§3) |
| No uXRCE-DDS agent or `px4_msgs` here | PX4 starts `uxrce_dds_client` on UDP 8888; nothing answers | thrust/torque setpoints cannot be flown; their timing is open (plan §9) |

Two authoring hazards nothing in `src/` catches: rotor sites that are not direct
children of `base_link` give a silently wrong lever arm (`MODELING_CONVENTIONS.md`
§2.2; the conversion script places them correctly), and a sidecar from another
vehicle with the same rotor count flies with the wrong thrust. Keep each sidecar
next to the MJCF it generated.

---

## 3. Next steps

1. **The coaxial `c_t` discount**, as its own step with its own flight. It moves
   the plant gain as well as the hover point: pick the factor in the sidecar,
   regenerate MJCF and airframe together (§7), re-fly Phase 5 and
   `fly_in_process.py steps`, and record both at plan phase 5.

**Strategy B** — baro, mag and GPS synthesized by the simulator, seeded — waits
until a use needs it: single-pair counterfactuals, sensor noise as a controlled
variable, or the IMU noise gap. Plan §3.3 has the evidence and what it takes.
**Aerodynamics** come after that, deliberately.

**The controller topology is decided and built.** In process, because the
research needs data ages tested at a *chosen* value, counterfactuals under
identical timing, stalls by design and batches faster than real time; out of
process all four could only be measured. The earlier reason for out of process,
"a slow solver cannot stall lockstep", was wrong: under lockstep a stall only
freezes PX4's clock. Phase 2's decisions, all built: strict lockstep as the only
regime; IMU → actuator one frame; estimates fetched after each answer at an age
of at least one frame, no `/proc`, no age 0; setpoints and commands behind a PING
barrier at least one frame ahead; strategy B later; EKF2's cadence phase left
unchosen. `sim/run_x8.py` in the private repo exposes the two PX4 delays.

Traps worth knowing before touching the loop or the API link (plan §3.2, §3.9
have the sources):

- **The receive thread blocks on simulated time** behind `SET_MESSAGE_INTERVAL`
  and `REQUEST_MESSAGE`: a barrier behind one hangs. The link refuses them.
- **PX4 drops the ACK to a new component's first message** when it races the
  registration; the silencing request sits between two PINGs for that reason.
- **No heartbeat of our own**: an `ONBOARD_CONTROLLER` heartbeat that lapses for
  `COM_OBC_LOSS_T` of PX4's clock raises "mission computer lost".
- **An answer does not prove the frame's estimate**, and **the barrier looks
  unnecessary when idle**: both only show under load, so test loaded.
- **PX4 starts answering 4–78 frames after it connects**; the fallback is needed.
- **`SET_ATTITUDE_TARGET` publishes its setpoint only in Offboard.**
- **PX4 cannot exit once simulated time stops** (plan §7): stop it first, or kill
  it, as `run_sitl.sh` and `fly_in_process.py` do.
- **A spinner process forked after `run()` inherits its SIGTERM handler**, which
  only stops a loop: kill load generators with SIGKILL.
- **Yaw read against the target carries EKF2's heading error**, 1–2° on the X8.

`../phase2_probes/`, next to the PX4 checkout, is the unversioned probe the design
came from; it keeps what was declined and strategy B's seeded sensors.

---

## 4. The platform

The X8's CAD export, filled sidecar, generated MJCF and generated airframe are
**private** and live in `~/DEV/kani_arm`; this repo holds the script, the
no-geometry sidecar template and the airframe template. Hub coordinates, inertia,
voxel counts and intrusion figures describe the vehicle as surely as the STLs,
so none of them goes into this repo — `urdf_to_mjcf.py --check-only` prints them.

`MODELING_CONVENTIONS.md` §7 records what the export carried, the motor fit
(thrust/weight 3.79, `km` 0.43× PX4's default), what is still a choice (`ω_idle`,
joint zero and sign, ESC wiring) and why the arm can reach into the discs inside
its mechanical limits. Intrusion is **reported, never blocked**, decided
2026-09-25: blocking belongs to the real vehicle's arm driver or companion
computer.

`vehicle.py`'s placeholder motors still warn on every load; that is now only the
quad. **The X8 loads with no warning**, and that silence is the check that the
sidecar's values reached `RotorModel`.

---

## 5. Which document wins

| Document | Owns | Wins on |
|---|---|---|
| `IMPLEMENTATION_PLAN.md` | PX4 v1.17 interface contract, phase design and exit criteria, troubleshooting, repo conventions, measured baselines | anything crossing the PX4 boundary: lockstep, message encoding, frames (§3.6 is the specification), geodetic origin |
| `MODELING_CONVENTIONS.md` | CAD → MJCF authoring, rotor parametrization, target platform, aerodynamic plans | anything about a model or a rotor: naming, frames *within* a model, thrust curve, X8 layout |
| `AGENTS.md` | state, gaps, next steps, how to work | "what now" |
| `README.md` | install, run, health check, side channel, in-process controller, ROS 2 integration | user-facing usage |

Where the first two overlap, the rule is scope, not seniority: §3.6's frame
conventions are the plan's and are binding on every model; the thrust curve is
`MODELING_CONVENTIONS.md` §2.5's and supersedes Phase 4's.

Progress belongs here, measured results in the plan at the phase that produced
them, the reasoning of a change in its commit. **This file stays short**: when a
step is done, its story leaves and one line of state stays.

Two templates own what no `.md` should restate:
[models/x8_arm.conversion.yaml.template](models/x8_arm.conversion.yaml.template)
(the sidecar schema, as comments) and
[px4/mujoco_x8.airframe.template](px4/mujoco_x8.airframe.template) (why a non-quad
airframe is shaped the way it is). Both are the no-geometry halves of private
files: edit the template, regenerate, never hand-edit a product.

---

## 6. Working conventions

- **All repository text in English**, code comments included — also when the
  conversation is not.
- **`source ../.venv/bin/activate` before any PX4 build**; PX4's cmake resolves its
  interpreter via `PATH`.
- **No `rclpy` / `ament` / `ros` imports in `src/`.**
- **Frame conversions live only in `frames.py`.**
- **Status line**: `unproven` stays 0, `brake` and `timeouts` flat after the boot;
  `ratio` alone proves nothing. Plan §7 diagnoses each.
- **Commit messages: short.** A title and a few lines saying what changed and where
  the numbers are; the documents own derivations. Worth a sentence in the commit
  and nowhere else: what an earlier version of the reasoning got wrong.
- **Never touch the private repo's git state**; propose changes there, and apply
  them only when asked.

### Keeping a session small

Past sessions reached 400–500K tokens of context. Most of it was not testing but
reading, rewriting and raw output. So:

- **Read by section, not by file.** `grep -n '^#' IMPLEMENTATION_PLAN.md`, then
  read the section that owns the question (§5). The plan is 1800 lines; nothing
  needs all of it.
- **Delegate lookups that return a conclusion.** PX4 source spelunking ("which
  paths end in `configure_stream_threadsafe()`?") and wide searches go to a
  subagent that reports file:line and the answer, not the files.
- **Flights print their report, nothing else.** `fly_in_process.py` is quiet by
  default; `acceptance` prints one table with PASS/FAIL. Never paste the
  simulator's log or PX4's into the session; grep it for the line that matters.
  For `run_sitl.sh` flights, keep the log in a file and read its last status line.
- **Settle the code on the cheapest check before the full matrix.** Unit tests,
  then `fly_in_process.py acceptance` on the quad, and only then the X8, the
  Phase 5 profiles and the steps. Re-running the whole matrix after a late code
  change is what doubled the flights last time.
- **Edit, do not rewrite.** Small exact-string edits keep a file out of the
  context; rewrite a file only when most of it changes.
- **Run tests quietly** (`python -m pytest -q | tail -3`), and read a failure's
  assertion, not its traceback.

---

## 7. Running each vehicle

The quad needs nothing beyond `README.md`. The X8 spans two repositories:

```sh
# Public repo, once per PX4 checkout:
./scripts/install_px4_files.sh --airframe ~/DEV/kani_arm/model/px4/22002_mujoco_x8
source ../.venv/bin/activate && make -C ../PX4-Autopilot px4_sitl_default

# From the private repo: PX4 via run_sitl.sh --px4-only, simulator and research
# controller in this process
python sim/run_x8.py

# From the public repo, on simulated time, PX4 in a throw-away rootfs:
X8="--model $HOME/DEV/kani_arm/model/mjcf/x8_arm.xml --rotors $HOME/DEV/kani_arm/model/mjcf/x8_arm.conversion.yaml --airframe 22002 --hover 0.2162"
python scripts/fly_in_process.py acceptance $X8
python scripts/fly_in_process.py steps $X8 --out /tmp/steps.npz
python scripts/fly_in_process.py arm-sweep $X8 --out /tmp/sweep_a.npz

# Without a controller, the arm over the side channel:
MUJOCO_SITL_ROTORS=$HOME/DEV/kani_arm/model/mjcf/x8_arm.conversion.yaml \
PX4_SYS_AUTOSTART=22002 ./scripts/run_sitl.sh -m $HOME/DEV/kani_arm/model/mjcf/x8_arm.xml
```

**The sidecar has no default.** Omitting it raises `RotorModel.spin has 4 entries
for 8 rotors`, so there is no silent fall back to the quad's motors.

After a sidecar edit, regenerate both products at once, which is what keeps
`CA_ROTOR*` and the rotor sites from disagreeing:

```sh
python scripts/urdf_to_mjcf.py ~/DEV/kani_arm/model/mjcf/x8_arm.conversion.yaml \
    --emit-airframe ~/DEV/kani_arm/model/px4/22002_mujoco_x8
```
