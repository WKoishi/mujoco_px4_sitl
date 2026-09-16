# mujoco_px4_sitl — Implementation Plan

MuJoCo as the physics backend for PX4 SITL, connected over the MAVLink HIL
interface. Target application: aerial manipulation (multirotor + robotic arm).

This document is the reference for the whole development effort. It records the
verified PX4 v1.17 interface contract, the phase order, and the exit criteria
for each phase. Update it when a decision changes; it is not a historical log.

---

## 1. Scope and constraints

**In scope**

- A standalone Python process that runs MuJoCo physics and speaks the PX4
  MAVLink simulator protocol (`simulator_mavlink` module).
- Sensor strategy A (see §3.3): the simulator provides IMU + ground-truth pose;
  PX4's own `sensor_*_sim` modules synthesize baro / mag / GPS.
- A launch entry point usable from a shell script or from a ROS 2 launch file
  via `ExecuteProcess`.
- A simulator-private side channel for arm commands and ground-truth state.

**Out of scope**

- No ROS 2 dependency in this repository. Not `rclpy`, not `ament`, not a
  colcon package layout. ROS 2 integration lives in a separate package that
  launches this process and talks to it over sockets.
- No modification of PX4 flight code. Configuration-only changes to PX4
  (airframe file + `.post` script) are acceptable and are tracked in §5.

**Rationale for Python**: PX4 SITL runs in lockstep, so the physics loop has no
real-time deadline — falling behind wall clock slows the simulation but cannot
corrupt it. That removes the main argument for C++. If a contact-rich
manipulation scenario later needs more throughput, the physics loop can be
moved to C++ behind the same socket boundary without protocol changes.

---

## 2. Architecture

```
 ROS 2 side (separate repo)          this repo                      PX4 SITL
 ┌──────────────────────┐   ┌─────────────────────────────┐   ┌──────────────┐
 │ arm controller node  │   │  mujoco_px4_sitl            │   │ px4 (posix)  │
 │ mission node         │   │                             │   │              │
 │                      │   │  ┌───────────────────────┐  │   │ simulator_   │
 │  uXRCE-DDS ──────────┼───┼──┼──> (PX4 direct, not   │  │   │  mavlink     │
 │                      │   │  │      through us)      │  │   │              │
 │  arm cmd / GT ───────┼───┼──┤  side channel (UDP)   │  │   │ sensor_baro_ │
 └──────────────────────┘   │  └───────────────────────┘  │   │  sim (20 Hz) │
                            │                             │   │ sensor_mag_  │
                            │  ┌───────────────────────┐  │   │  sim (50 Hz) │
                            │  │ MuJoCo mj_step        │  │   │ sensor_gps_  │
                            │  │ rotor thrust model    │  │   │  sim (8 Hz)  │
                            │  │ IMU sampling          │  │   │              │
                            │  └───────────────────────┘  │   │ ekf2         │
                            │            │                │   │              │
                            │  TCP :4560 (we are SERVER)  │   │              │
                            └─────────────┼───────────────┘   └──────┬───────┘
                                          └── HIL_SENSOR ────────────┤
                                              HIL_STATE_QUATERNION ──┤
                                          ┌── HIL_ACTUATOR_CONTROLS ─┘
```

Two independent transports, deliberately kept separate:

| Channel | Direction | Transport | Payload |
|---|---|---|---|
| PX4 HIL | bidirectional | TCP `:4560`, we listen | `HIL_SENSOR`, `HIL_STATE_QUATERNION`, `HIL_ACTUATOR_CONTROLS` |
| Side channel | bidirectional | UDP (configurable port) | arm joint commands, arm state, full ground truth |

The side channel is our own schema. It is not MAVLink and PX4 never sees it.

---

## 3. PX4 interface contract (verified against PX4 v1.17.0, `d6f12ad1c4`)

Every claim in this section was read out of the PX4 tree. File references are
relative to `PX4-Autopilot/`. Do not change these details without re-reading the
cited code.

### 3.1 Transport and connection roles

`px4-rc.mavlinksim` starts `simulator_mavlink start -c $((4560+px4_instance))`.
With `-c`, PX4 calls `connect()` in a retry loop
(`src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp:1154-1181`).

**PX4 is the TCP client. We must be the TCP server and bind first.** PX4 retries
every 500 µs until we accept, so we may start before or after PX4.

`TCP_NODELAY` is set on the PX4 side; set it on ours too. MAVLink v2 framing;
PX4 sends with `MAV_SYS_ID` / `MAV_COMP_ID` (default 1/1).

### 3.2 Lockstep timing

Lockstep is enabled in `px4_sitl_default` (`boards/px4/sitl/sitl.cmake:10-12`).
The contract (`SimulatorMavlink.cpp:503-548`, `:1028-1070`):

1. We send `HIL_SENSOR` with `id == 0`.
2. PX4 calls `px4_clock_settime(CLOCK_MONOTONIC, imu.time_usec)` — **our IMU
   timestamp is PX4's clock**. It must be strictly monotonic with stable `dt`.
3. PX4 registers a lockstep component on the first `id == 0` message, runs its
   pipeline, then calls `px4_lockstep_progress()`.
4. PX4's sender thread waits on `actuator_outputs_sim`, then emits
   `HIL_ACTUATOR_CONTROLS`.

Our loop therefore is: `mj_step` → send `HIL_SENSOR` → block on
`HIL_ACTUATOR_CONTROLS` → apply controls → repeat. Blocking on the actuator
message is what keeps the two processes in step. There is no separate
handshake and no `SYSTEM_TIME` exchange to implement.

`HIL_ACTUATOR_CONTROLS.flags` bit 0 is set when PX4 was built with lockstep
(`SimulatorMavlink.cpp:134-136`). Check it at startup and warn loudly if it is
clear, because the loop above will then deadlock — a `nolockstep` PX4 build
needs a timeout-based loop instead.

### 3.3 Sensor split (strategy A)

`CONFIG_COMMON_SIMULATION=y` compiles `sensor_baro_sim`, `sensor_mag_sim`,
`sensor_gps_sim`, `sensor_airspeed_sim` into `px4_sitl_default`
(`src/modules/simulation/Kconfig:6-14`). They subscribe to ground-truth uORB
topics, not to anything gz-specific:

| Module | Subscribes | Publishes | Rate |
|---|---|---|---|
| `sensor_baro_sim` | `vehicle_global_position_groundtruth` | `sensor_baro` | 20 Hz |
| `sensor_mag_sim` | `vehicle_attitude_groundtruth`, `vehicle_global_position_groundtruth` | `sensor_mag` | 50 Hz |
| `sensor_gps_sim` | `vehicle_global_position_groundtruth`, `vehicle_local_position_groundtruth` | `sensor_gps` | 8 Hz |

`simulator_mavlink` publishes all four ground-truth topics from
`HIL_STATE_QUATERNION` (`SimulatorMavlink.hpp:250-253`,
`SimulatorMavlink.cpp:551-640`). So the chain closes without us synthesizing
baro/mag/GPS ourselves.

**Two things are not wired up by default and are ours to fix (§5):**

- `px4-rc.mavlinksim` does not start these modules. Only `px4-rc.gzsim` and
  `px4-rc.sihsim` do.
- Each module is gated on a parameter defaulting to `0`: `SENS_EN_BAROSIM`,
  `SENS_EN_MAGSIM`, `SENS_EN_GPSSIM` (`.../sensor_*_sim/parameters.c`). The gate
  is evaluated by the startup script, not inside the module, so the parameter
  must be set *and* the `start` command must be issued.

### 3.4 `HIL_SENSOR.fields_updated` bitmask

From `SimulatorMavlink.hpp:89-95`. Per-sensor, independently checked:

```
ACCEL      = 0b0000000000111  = 0x0007
GYRO       = 0b0000000111000  = 0x0038
MAG        = 0b0000111000000  = 0x01C0
DIFF_PRESS = 0b0010000000000  = 0x0400
BARO       = 0b1101000000000  = 0x1A00
```

Strategy A sends `ACCEL | GYRO` = `0x003F` only. Unset groups are skipped
entirely, so leaving mag/baro fields at zero is safe and supported. Note
`temperature` is only consumed when the `BARO` bit is set — under strategy A the
IMU temperature stays at PX4's default, which is fine.

This bitmask is the migration seam toward strategy B: to take over one sensor
later, set its bit and stop starting the corresponding `sensor_*_sim` module.
Nothing else changes.

### 3.5 `HIL_ACTUATOR_CONTROLS` scaling

`pwm_out_sim` normalizes before publishing (`.../pwm_out_sim/PWMSim.cpp:60-98`):

- Non-reversible motor functions (`Motor1..MotorMax`): **`[0, 1]`**
- Everything else, including servos: **`[-1, 1]`**
- Disarmed outputs stay at `0`; `SimulatorMavlink::actuator_controls_from_outputs`
  zeroes the whole array unless armed (`SimulatorMavlink.cpp:124-131`).

`msg.mode` carries `mode_flag_armed = 128`. Use it to gate thrust application,
and treat "armed bit clear" as "all actuators zero" rather than trusting the
payload.

Values are normalized commands, not PWM and not RPM. Mapping them to rotor
thrust is our model's job (§6, phase 4).

### 3.6 Frames and conventions

Verified empirically with MuJoCo 3.13.0 in this workspace:

- A resting free body reads `[0, 0, +9.81]` on a body-aligned `accelerometer`
  site; in free fall it reads `[0, 0, 0]`. MuJoCo's accelerometer is **specific
  force**, which is exactly what PX4's accel expects. No gravity term must be
  added by hand — only a frame rotation.
- `freejoint` layout is `qpos = [x, y, z, qw, qx, qy, qz]` (world frame, z-up),
  `qvel = [vx, vy, vz, wx, wy, wz]`.

Required conversions, all in one module with one test file (§6, phase 3):

| Quantity | MuJoCo | PX4 |
|---|---|---|
| World frame | ENU-like, z-up | NED |
| Body frame | FLU (x fwd, y left, z up) | FRD (x fwd, y right, z down) |
| Attitude | quaternion world→body, `[w,x,y,z]` | quaternion NED→FRD, `[w,x,y,z]` |
| Position | metres, local | lat/lon/alt (deg×1e7, mm) in `HIL_STATE_QUATERNION` |

Body FLU→FRD is a 180° rotation about x: negate the y and z components of every
body-frame vector. World ENU→NED: `(x_n, y_e, z_d) = (y, x, -z)`. Geodetic
conversion needs a fixed home reference (`PX4_HOME_LAT` / `PX4_HOME_LON` /
`PX4_HOME_ALT` semantics) and a local-tangent projection matching PX4's
`MapProjection`.

Getting this wrong is the most likely cause of EKF2 divergence, ahead of any
noise-model detail. Phase 3 exists to isolate it.

---

## 4. Repository layout

```
mujoco_px4_sitl/
├── IMPLEMENTATION_PLAN.md      this file
├── README.md                   quickstart, kept short
├── pyproject.toml              deps: mujoco>=3.13, numpy, pymavlink
├── src/mujoco_px4_sitl/
│   ├── __init__.py
│   ├── main.py                 CLI entry point (script-launchable)
│   ├── config.py               dataclass config, CLI + file override
│   ├── transport.py            TCP server, MAVLink framing, reconnect
│   ├── hil.py                  HIL_SENSOR / HIL_STATE_QUATERNION encode,
│   │                           HIL_ACTUATOR_CONTROLS decode
│   ├── frames.py               MuJoCo <-> PX4 frame + geodetic conversion
│   ├── sim.py                  MjModel/MjData ownership, step, sensor read
│   ├── vehicle.py              rotor thrust/torque model, actuator mapping
│   ├── loop.py                 lockstep orchestration
│   ├── sidechannel.py          UDP arm-command / ground-truth server
│   └── viewer.py               optional mujoco.viewer, off by default
├── models/
│   ├── quad_x.xml              phase 1-5 airframe
│   └── quad_x_arm.xml          phase 7 aerial manipulator
├── px4/
│   ├── 4600_mujoco_quad        PX4 airframe file (copied into PX4 tree)
│   └── 4600_mujoco_quad.post   starts the sensor_*_sim modules
├── scripts/
│   ├── run_sitl.sh             launches PX4 + simulator together
│   └── install_px4_files.sh    copies px4/ into the PX4 tree
└── tests/
    ├── test_frames.py          conversion round-trips, analytic cases
    ├── test_hil.py             message encode/decode against pymavlink
    └── test_vehicle.py         rotor model: hover thrust, torque balance
```

`viewer.py` stays optional and off by default: under lockstep the viewer's
event loop must not gate the physics loop, and headless CI must not need GL.

---

## 5. PX4-side configuration

Two files, installed into the PX4 tree by `scripts/install_px4_files.sh`. Keep
them in this repo so the PX4 checkout stays a clean upstream tree apart from a
documented copy step.

**`px4/4600_mujoco_quad`** — airframe file, modelled on
`ROMFS/px4fmu_common/init.d-posix/airframes/10016_none_iris`. Because the
selector in `px4-rc.simulator` falls through to `px4-rc.mavlinksim` for anything
that is not sihsim / gz / jmavsim, a plain airframe id gets us the right
simulator with no script edit. Contents:

- `. ${R}etc/init.d/rc.mc_defaults`
- `CA_*` rotor geometry matching `models/quad_x.xml` exactly — same arm length,
  same rotor order, same spin directions. A mismatch here shows up as a slow
  yaw drift or a roll/pitch cross-coupling that is easy to misread as an EKF
  problem.
- `PWM_MAIN_FUNC1..4 = 101..104`
- `param set-default SENS_EN_BAROSIM 1`, `SENS_EN_MAGSIM 1`, `SENS_EN_GPSSIM 1`
- `param set-default IMU_INTEG_RATE 250` (already the SITL default, set it
  explicitly so the simulator's IMU rate and PX4's expectation are stated in
  one place)

**`px4/4600_mujoco_quad.post`** — sourced near the end of `rcS`, after
`px4-rc.simulator` has started `simulator_mavlink`:

```sh
sensor_baro_sim start
sensor_mag_sim start
sensor_gps_sim start
```

Ordering is not critical under lockstep (PX4's clock does not advance until our
first `HIL_SENSOR` arrives), but `.post` is the documented hook and needs no
upstream file edits.

Launch shape: `PX4_SYS_AUTOSTART=4600 ./build/px4_sitl_default/bin/px4`.
Airframe id 4600 is unused upstream; verify with a glob over
`init.d-posix/airframes/` before committing.

---

## 6. Phases

Each phase ends with a runnable artifact and a check that can fail. Do not start
a phase before its predecessor's exit criteria pass — a frame bug found in phase
3 is a ten-minute fix, and the same bug found in phase 5 looks like an EKF
tuning problem.

### Phase 0 — Environment and skeleton

- `pyproject.toml` with pinned deps. `pymavlink` is **not yet installed** in the
  workspace venv; add it (pin the exact version).
- Build PX4 once: `make px4_sitl_default`. Confirm
  `build/px4_sitl_default/bin/px4` exists and that
  `sensor_baro_sim`/`sensor_mag_sim`/`sensor_gps_sim` appear in the built
  command list.
- Install the two PX4 files from §5.

**Exit**: `python -m mujoco_px4_sitl --help` runs; PX4 boots with
`PX4_SYS_AUTOSTART=4600` and logs `Waiting for simulator to accept connection on
TCP port 4560`.

### Phase 1 — Minimal closed loop, physics stubbed out

Prove the protocol before trusting any dynamics. `sim.py` returns a hardcoded
level-hover state; no `mj_step` yet.

- TCP server on 4560, accept one client.
- Send `HIL_SENSOR` (`fields_updated = 0x3F`) at 250 Hz of *simulated* time with
  a synthetic monotonic clock, `id = 0`, accel `[0, 0, -9.81]` in FRD (level and
  stationary — see §3.6), gyro zero.
- Send `HIL_STATE_QUATERNION` at 100 Hz, identity attitude, fixed home position.
- Receive and log `HIL_ACTUATOR_CONTROLS`; assert `flags & 1` (lockstep).

**Exit**: `commander status` shows no sensor timeouts; `listener sensor_baro`,
`listener sensor_mag`, `listener sensor_gps` all produce data — this is the
single most important check in the plan, because it is the runtime confirmation
of §3.3, which so far is established only by source reading. `ekf2 status`
reports the attitude filter running. Simulated time advances in step with PX4.

### Phase 2 — Real MuJoCo state, open loop

Replace the stub with `models/quad_x.xml` and real `mj_step`. Actuator inputs
still ignored; hold the vehicle in place with a weld or by resetting state each
step, so ground truth moves in a known way.

- IMU from MuJoCo `accelerometer` + `gyro` sensors on a body-fixed site.
- Simulation clock derived from `mj_data.time`, not wall clock.
- IMU at `IMU_INTEG_RATE` (250 Hz); physics `dt` an integer divisor of it
  (e.g. 1 kHz physics, IMU every 4th step). Record the ratio in `config.py`.

**Exit**: injecting a known attitude in MuJoCo produces the matching attitude in
`vehicle_attitude_groundtruth` (compare via `listener`), for at least roll,
pitch, and yaw taken one at a time.

### Phase 3 — Frames, isolated and tested

`frames.py` plus `tests/test_frames.py`. Written as a standalone phase because
this is where the expensive bugs live.

- ENU↔NED, FLU↔FRD, quaternion conversion, geodetic projection.
- Property tests: round-trip identity; a 90° yaw in MuJoCo is a 90° yaw of the
  correct sign in PX4; a level stationary vehicle yields accelerometer
  `[0, 0, -9.81]` in FRD (MuJoCo reads `[0, 0, +9.81]` in its z-up site frame;
  the FLU→FRD y/z negation flips the sign, and specific force at rest is the
  negative of the gravity vector — PX4's convention is accel z ≈ -9.8 when
  level, while the *gravity vector itself* in FRD is `+z`. Keep the two
  distinct; conflating them is exactly the phase-3 bug this test catches).
- Cross-check the geodetic projection against PX4's `MapProjection` behaviour
  using a handful of hardcoded reference points.

**Exit**: `pytest tests/test_frames.py` green, and the phase-2 attitude check
passes for all three axes plus one combined 45°/45° case.

### Phase 4 — Rotor model and actuator mapping

- Map `HIL_ACTUATOR_CONTROLS.controls[0..3]` ∈ `[0,1]` to thrust and reaction
  torque per rotor. Start with quadratic thrust vs. normalized command, first
  order motor lag, and a `CA_ROTOR*_KM`-consistent torque coefficient.
- Apply as MuJoCo forces on rotor sites; keep the model's rotor order identical
  to the `CA_ROTOR*` indices in the airframe file.
- Calibrate so that hover sits near mid-stick: total thrust at command 0.5
  should be close to vehicle weight.

**Exit**: `tests/test_vehicle.py` confirms hover thrust equals weight within
tolerance and that yaw torque sums to zero with all four rotors at equal
command. Manually commanding all rotors to hover value keeps the vehicle
airborne and roughly level for several seconds.

### Phase 5 — Full flight validation

First phase where PX4 is genuinely flying the MuJoCo vehicle.

- Arm, takeoff, position hold, a square in Offboard or Mission, land.
- Compare `vehicle_local_position` (EKF2) against
  `vehicle_local_position_groundtruth` from the log.

**Exit**: EKF2 converges within ~10 s of boot; position error stays under a few
tens of centimetres in hover; a full takeoff → square → land completes with no
failsafe. Keep the resulting ulog as the regression baseline.

If EKF2 misbehaves here, work down §7 in order before touching EKF2 parameters.

### Phase 6 — External API and launch story

- `scripts/run_sitl.sh`: starts PX4 and the simulator, plumbs ports, forwards
  signals, exits cleanly on Ctrl-C.
- `sidechannel.py`: UDP server with a versioned message schema (JSON first;
  swap for a packed binary format only if profiling says so). Publishes ground
  truth, accepts arm joint commands.
- Config via CLI flags and env vars, no hardcoded paths. Multi-instance support:
  derive the HIL port as `4560 + instance`, matching PX4's convention.
- Document the ROS 2 integration pattern in `README.md` — `ExecuteProcess` plus a
  thin bridge node in the *other* repo — but write no ROS 2 code here.

**Exit**: a plain shell script brings up a flying vehicle; an external process
reads ground truth over the side channel without touching this repo's internals.

### Phase 7 — Aerial manipulator

- `models/quad_x_arm.xml`: multirotor plus serial arm, actuated joints with
  realistic limits, torque limits, and mass.
- Arm joints driven from the side channel (position or torque, decide when the
  arm's controller design is known).
- Expected physics coupling: the arm moves the composite CoM and adds reaction
  torques on the base. PX4 will see this as a disturbance, which is the point.
- Investigate whether the arm needs its own control rate distinct from the
  physics rate.

**Exit**: arm motion during hover produces a measurable, bounded attitude
disturbance that PX4 rejects without losing position lock. Ground truth for
both base and arm is available externally.

---

## 7. EKF2 divergence checklist

In observed likelihood order. Work top to bottom; do not start tuning EKF2
parameters until every item above the one you suspect has been excluded.

1. **Body frame sign error** — FLU vs FRD. Symptom: instant divergence on arming,
   or control that fights itself. Check that a level stationary vehicle sends
   accel `[0, 0, -9.81]` (FRD), and that `listener sensor_accel` in PX4 agrees.
2. **World frame swap** — ENU vs NED. Symptom: north/east transposed, or
   altitude inverted; position control drives the wrong way.
3. **Non-monotonic or jittery IMU timestamps** — this *is* PX4's clock under
   lockstep. Symptom: erratic filter behaviour, lockstep stalls, log gaps.
   Assert monotonicity in code, not just in review.
4. **Wrong IMU rate** — must match `IMU_INTEG_RATE` (250 Hz).
5. **Quaternion convention** — `[w,x,y,z]` on both sides, and world→body
   direction, not body→world. Off-by-conjugate looks like inverted attitude.
6. **`HIL_STATE_QUATERNION` rate too low or missing fields** — baro/mag/GPS all
   derive from it. If `sensor_gps` is stale, check that local-position ground
   truth carries velocity, since `sensor_gps_sim` reads `vx/vy/vz` from it.
7. **Rotor geometry mismatch** between `CA_ROTOR*` and the MuJoCo model.
   Symptom: slow yaw drift, or roll/pitch coupling. Frequently misdiagnosed as
   an estimator fault.
8. Only now: EKF2 parameters.

Useful during bring-up: `listener <topic>`, `ekf2 status`, `commander status`,
`uorb top`. For a suspected sensor-synthesis problem, run the same airframe under
gz and diff the topic behaviour — keeping that comparison available is a
deliberate benefit of strategy A.

---

## 8. Conventions

- All repository text in English, including code comments.
- Python ≥ 3.12 (workspace venv is 3.12.3, matching ROS 2 Jazzy, so a future
  in-process `rclpy` bridge stays possible without a rebuild).
- Type hints on public functions; `numpy` for all vector math; no `ros`, no
  `rclpy`, no `ament` imports anywhere in `src/`.
- Units: SI internally. Convert to MAVLink's scaled integers only at the encode
  boundary in `hil.py`.
- One module owns each concern. Frame conversion happens only in `frames.py`;
  if a rotation appears elsewhere, that is a bug.
- Tests run headless and without PX4 where possible. Protocol and frame tests
  must not require a PX4 build.

## 9. Open questions

Resolve when the phase that needs them arrives, not before.

- Arm control interface: joint position targets vs. torque. Depends on the
  manipulation controller design.
- Physics rate for contact-rich manipulation: 1 kHz may not be enough; measure
  before deciding, and note that raising it costs wall-clock time only, not
  correctness.
- Whether the aerial manipulator eventually needs a custom `CA_*` allocation or
  benefits from PX4's existing disturbance rejection alone.
