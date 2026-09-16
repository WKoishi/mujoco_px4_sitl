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
corrupt it. (The hazard runs the other way: getting *ahead* of PX4 does corrupt
it. See §3.2 — this is a property of the loop design, not of the language.) That
removes the main argument for C++. If a contact-rich
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
| PX4 HIL | bidirectional | TCP `:4560`, we listen | out: `HIL_SENSOR`, `HIL_STATE_QUATERNION`; in: `HIL_ACTUATOR_CONTROLS`, plus `HEARTBEAT` / `COMMAND_LONG` we discard (§3.1) |
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

PX4's receive thread polls the socket with a **wall-clock** 1000 ms timeout and
logs `PX4_ERR("poll timeout ...")` on expiry (`SimulatorMavlink.cpp:1217-1221`).
This is a different call from the sender thread's simulated-time
`px4_poll(..., 100)` discussed in §3.2 — do not conflate them. If we stall the
wall clock for a second (a breakpoint, a blocking write), PX4 spams this error;
it is benign and clears on its own, but it looks like a fault during bring-up.

Immediately after connecting, PX4 sends two messages we did not ask for and must
not choke on:

- `HEARTBEAT`, from the sender thread before it enters its poll loop
  (`SimulatorMavlink.cpp:1084-1093`).
- `COMMAND_LONG` with `MAV_CMD_SET_MESSAGE_INTERVAL`, `param1 =
  MAVLINK_MSG_ID_HIL_STATE_QUATERNION`, `param2 = 5000` µs
  (`SimulatorMavlink.cpp:1073-1082`, called from `run()` at `:1212`).

Their **order is not guaranteed**: the `HEARTBEAT` is sent by the sender thread
(`sending_trampoline` → `send_heartbeat()`), the `COMMAND_LONG` by the `run()`
thread *after* `pthread_create` has already launched that sender
(`SimulatorMavlink.cpp:1206-1212`). Do not write a parser that expects one
before the other, and do not gate anything on either arriving.

Neither needs a reply — PX4 ignores whether we honour the interval. But the
`param2 = 5000` is PX4 telling us it wants ground truth at **200 Hz**. Note 200 Hz
is **not** an integer divisor of the 250 Hz IMU rate (§5), so it cannot be
expressed as "every Nth IMU frame". Since nothing in PX4 checks the interval,
**send `HIL_STATE_QUATERNION` on every IMU frame, i.e. at 250 Hz**, and treat 200 Hz
as a floor rather than a target. Faster is free and keeps one cadence in the loop.
Our parser must accept and discard any other inbound message id without
desynchronising the MAVLink framing.

### 3.2 Lockstep timing

Lockstep is enabled in `px4_sitl_default` (`boards/px4/sitl/sitl.cmake:10-12`).
The contract (`SimulatorMavlink.cpp:503-548`, `:1028-1070`):

1. We send `HIL_SENSOR` with `id == 0`.
2. PX4 calls `px4_clock_settime(CLOCK_MONOTONIC, imu.time_usec)` — **our IMU
   timestamp is PX4's clock**. It must be strictly monotonic with stable `dt`.
3. PX4 registers a lockstep component on the first `id == 0` message, runs its
   pipeline, then calls `px4_lockstep_progress()`.
4. PX4's sender thread waits on `actuator_outputs_sim`, then calls
   `px4_lockstep_wait_for_components()` and emits `HIL_ACTUATOR_CONTROLS`.

**PX4's boot blocks on our first `HIL_SENSOR`.** Under lockstep,
`simulator_mavlink start` does not return until `_has_initialized` is set, and
that happens in `handle_message_hil_sensor()` on the first `id == 0` message
(`SimulatorMavlink.cpp:1669-1681`, `:538-548`). `rcS` therefore stops inside
`px4-rc.simulator` until we connect *and* send. Everything after that line —
`commander`, `ekf2`, `navigator`, and the `.post` script that starts the
`sensor_*_sim` modules (§5) — boots while our loop is already running and already
advancing PX4's clock. That window is the least supervised part of the whole
design: nothing in PX4 is yet in a position to push back on us. The pacing rules
below apply to it first, not last.

**Lockstep here is one-directional, and that asymmetry drives the loop design.**

*PX4 cannot run ahead of us.* Its entire notion of time comes from our IMU
timestamps, so with no `HIL_SENSOR` there is no time and nothing in PX4
progresses. This half is solid.

*But we can run ahead of PX4, arbitrarily far.* `px4_clock_settime` does **not**
block. `LockstepScheduler::set_absolute_time()` writes `_time_us`, broadcasts the
condition variables of every timed wait that has now expired, and returns
immediately — it never waits for the woken threads to finish
(`lockstep_scheduler.cpp:51-98`). `px4_lockstep_progress()` likewise only sets a
bit and posts a semaphore (`lockstep_components.cpp:99-125`). The one call that
blocks, `px4_lockstep_wait_for_components()`, runs on PX4's **sender** thread
(`SimulatorMavlink.cpp:1064`) and gates only the outgoing
`HIL_ACTUATOR_CONTROLS`. **Nothing on PX4's receive path applies back-pressure to
us.**

So if we never wait, simulated time advances as fast as our CPU allows and the
only back-pressure is the TCP socket buffer filling — hundreds of messages later.
That is not harmless. `update_sensors()` publishes accel/gyro through the
simulated FIFO path, and `_last_gyro_fifo.dt` is computed straight from the delta
between our timestamps (`SimulatorMavlink.cpp:263-268`). If ekf2 falls behind,
samples are overwritten and dropped, which presents exactly as the estimator
divergence §7 is written to diagnose.

**Conclusion: the loop needs a pacer that does not depend on PX4 at all, plus a
brake that does.** The two failure modes are distinct and both real: blocking from
the first frame deadlocks at startup (below); never blocking decouples us from
PX4's pipeline — during flight *and* during the boot window above. Wall clock is
the pacer, because it works before PX4 has booted; the actuator stream is only the
brake.

**`HIL_ACTUATOR_CONTROLS` is not an acknowledgement of `HIL_SENSOR`.** It is
published only when the control chain produces new outputs:

```
HIL_SENSOR → vehicle_angular_velocity → mc_rate_control
    → vehicle_torque_setpoint → ControlAllocator → actuator_motors
    → pwm_out_sim → actuator_outputs_sim → HIL_ACTUATOR_CONTROLS
```

Every link can decline to publish. `pwm_out_sim` only publishes when
`num_control_groups_updated > 0` (`PWMSim.cpp:65-67`), and `mc_rate_control` only
publishes setpoints when `flag_control_rates_enabled` is set
(`MulticopterRateControl.cpp:189`). Once the chain is warm the rate is close to
the IMU rate, but it is a consequence of the control cascade, not a guarantee,
and it does not hold during the first moments after boot.

**Therefore: do not build the loop as "send `HIL_SENSOR`, block until
`HIL_ACTUATOR_CONTROLS` arrives" *from the first frame*.** That deadlocks at
startup. Before Commander has published `vehicle_control_mode`, the chain above is
silent, so no actuator message is produced. If we are blocked waiting for one, we
send no `HIL_SENSOR`, so simulated time does not advance — and PX4's escape
hatches cannot save us:

- the sender thread's `px4_poll(..., 100)` timeout
  (`SimulatorMavlink.cpp:1044`) is measured in *simulated* time and never fires;
- `ControlAllocator`'s `ScheduleDelayed(50_ms)` backup is not merely inert, it is
  **compiled out entirely** in a lockstep build — both call sites sit under
  `#ifndef ENABLE_LOCKSTEP_SCHEDULER` with the comment "Backup schedule would
  interfere with lockstep" (`ControlAllocator.cpp:98-100`, `:312-316`).

Circular wait, permanent hang.

The loop has one regime, not two. Wall clock paces it; the actuator stream bounds
how far ahead of PX4 it may get:

```
MAX_LEAD_FRAMES = 8          # IMU frames we may have outstanding before blocking
frames_since_ack = 0
controls = zeros             # last-known-good
t_wall_next = now()

loop:
  mj_step until the next IMU boundary
  send HIL_SENSOR (id 0)           # this advances PX4's clock
  send HIL_STATE_QUATERNION        # same cadence, see below
  frames_since_ack += 1

  drain the socket non-blocking
  on each HIL_ACTUATOR_CONTROLS: latch controls, frames_since_ack = 0

  if frames_since_ack >= MAX_LEAD_FRAMES:
      # brake: PX4's pipeline is behind, or silent. Wait, but never forever.
      block for a fresh HIL_ACTUATOR_CONTROLS with a WALL-CLOCK timeout
      on arrival: latch controls, frames_since_ack = 0
      on timeout:  log once per N, keep previous controls,
                   frames_since_ack = 0 and continue
                   # never fatal, never expressed in simulated time

  apply the latched controls to MuJoCo   # last-known-good on a quiet frame

  # pacer: independent of PX4, so it also governs the boot window above
  t_wall_next += imu_dt / speed_factor
  sleep until t_wall_next        # skip if already past: we are CPU-bound, not ahead
```

Why this shape rather than a startup/steady-state latch:

- **The pacer works before PX4 exists.** During the boot window it is the *only*
  thing bounding us, and a latch that free-runs until the first actuator message
  leaves exactly that window unbounded (§3.2 above).
- **The brake does not depend on a 1:1 actuator/sensor ratio.** As established
  above, `HIL_ACTUATOR_CONTROLS` is not an acknowledgement, so "wait for a fresh
  one every frame" degrades to one frame per timeout whenever the ratio slips.
  With a hundreds-of-ms timeout on a 250 Hz loop that is a ~100× slowdown that
  logs as "non-fatal" — a silent failure, and the worst kind. Allowing a few
  outstanding frames decouples back-pressure from PX4's publish rate.
- **Resetting `frames_since_ack` on timeout is deliberate.** It hands pacing back
  to the wall clock during a genuine PX4 silence (disarm, mode transition). We
  cannot outrun PX4 while real-time paced, so the FIFO-drop hazard does not apply
  there; the brake exists for when PX4 is *slow*, not when it is *quiet*.

`MAX_LEAD_FRAMES` and the timeout are both tunables, not contract. Set the timeout
generously — hundreds of milliseconds of wall clock — since its only job is to
break a deadlock. If physics is slower than real time (contact-rich manipulation),
the `sleep` never fires and the bounded lead becomes the binding constraint, which
is the correct degradation.

Reusing the previous frame's controls on a quiet frame is correct: they are a held
setpoint.

**What jMAVSim actually does**, since it is the reference implementation and the
detail is easy to misread: its main loop runs on a **wall-clock** fixed-rate
executor, `scheduleAtFixedRate(this, 0, sleepInterval / speedFactor / checkFactor)`
with `sleepInterval = 1e6 / 250` (`jMAVSim/src/me/drton/jmavsim/Simulator.java:114,390-391`).
That timer is present in both phases. `gotHilActuatorControls` is a one-way latch
set on the first actuator message and cleared only in `reset()`
(`MAVLinkHILSystem.java:26,54,292`), and the
`if (!hilSystem.gotHilActuatorControls()) advanceTime()` in `run()`
(`Simulator.java:515-524`) removes the *need to wait* for an actuator message
during startup — it does not remove the timer. So bootstrap is 250 Hz real-time
paced, not free-running. In steady state time advances on actuator receipt inside
`handleMessage()` (`MAVLinkHILSystem.java:73`), but physics and sending still
happen on the timer tick, skipped via `needsToPause = (lastTimeRan == now)` when
time has not moved. Neither phase is "free-run", and neither is "block until
actuator": wall-clock pacing is the constant, and the actuator stream only gates
whether time advances. The loop above keeps that constant and makes the gate a
bounded lead instead of a per-frame wait.

Beyond the two unsolicited startup messages in §3.1, there is no handshake and no
`SYSTEM_TIME` exchange to implement.

`HIL_ACTUATOR_CONTROLS.flags` bit 0 is set when PX4 was built with lockstep
(`SimulatorMavlink.cpp:134-136`). Check it on the first actuator message and warn
loudly if it is clear: a `nolockstep` build does not take its clock from us, so
the IMU cadence must then be paced against wall clock instead.

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

- `px4-rc.mavlinksim` does not start these modules. Only `px4-rc.sihsim` starts
  all of baro / mag / GPS (`px4-rc.sihsim:18-32`); `px4-rc.gzsim` starts only mag
  and airspeed (`px4-rc.gzsim:198-205`), because gz supplies baro and GPS itself.
  So there is no existing posix script we can copy wholesale — sihsim is the
  closest model.
- Each module is gated by a `param compare -s SENS_EN_*SIM 1` test **in the
  startup script**, not inside the module. `SENS_EN_BAROSIM`, `SENS_EN_MAGSIM`,
  `SENS_EN_GPSSIM` all default to `0` (`.../sensor_*_sim/parameters.c`). None of
  the three modules reads its own `SENS_EN_*SIM`: their `DEFINE_PARAMETERS` blocks
  list only `SIM_BARO_OFF_P/T`, `SIM_MAG_OFFSET_X/Y/Z` and `SIM_GPS_USED`
  respectively (`SensorBaroSim.hpp:88-91`, `SensorMagSim.hpp:89-93`,
  `SensorGpsSim.hpp:85-87`). Consequence: because our `.post` issues `start`
  unconditionally (§5), setting these parameters is *not* required for the modules
  to run. We set them anyway so that `param show SENS_EN_*` reflects reality and so
  the airframe stays consistent with how sihsim and gzsim express the same intent.

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
`temperature` is only consumed when the `BARO` bit is set
(`SimulatorMavlink.cpp:191-196`) — under strategy A the IMU temperature stays at
PX4's default of `0`, which is fine: thermal calibration is off by default and no
consumer in this path reads it.

`id` is a MAVLink v2 **extension** field. v2 zero-trimming means a truncated
message decodes to `id == 0`, which is what we want anyway, so this is safe either
way — but be explicit about it rather than relying on the accident.

**Sensor id 0 goes through PX4's simulated FIFO path**, not the plain update path
(`SimulatorMavlink.cpp:246-270`). Values are quantised: gyro by
`radians(2000/32768)` ≈ 0.001 °/s per LSB, accel similarly. Irrelevant for flight,
but any test that compares a PX4-side `listener sensor_gyro` reading against the
exact value we sent needs a tolerance, not equality.

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

**Model-authoring constraints.** MuJoCo fixes only that the world is z-up.
Everything else below is a rule `models/*.xml` must obey, not a property we can
rely on. Two rules, both load-bearing:

1. **Body frame is FLU** — x forward, y left, z up, with the IMU site aligned to
   it. Body axis orientation is entirely the model author's choice.
2. **World frame is ENU** — world **+x is east**, +y is north, +z is up. MuJoCo
   has no opinion here beyond z-up; "ENU" is our choice. It is what makes
   `q_ned_enu` below correct, and it is where the +90° yaw offset in the attitude
   table comes from. A model authored as x-north (NWU-style, which some bridges
   use) needs a *different* `q_ned_enu` and would silently produce a 90° heading
   error with the conversion below.

Violating either silently invalidates every conversion below; re-check both
whenever a model is added or edited.

Required conversions, all in one module with one test file (§6, phase 2):

| Quantity | MuJoCo | PX4 |
|---|---|---|
| World frame | ENU (x east, y north, z up — our model convention, see above) | NED |
| Body frame | FLU (x fwd, y left, z up) | FRD (x fwd, y right, z down) |
| Attitude | quaternion body→world (FLU→ENU), `[w,x,y,z]` | quaternion body→world (FRD→NED), `[w,x,y,z]` |
| Position | metres, local | lat/lon/alt (deg×1e7, mm) in `HIL_STATE_QUATERNION` |
| Angular velocity | body rates in FLU, rad/s | body rates in FRD, rad/s — **negate y and z** |

**Angular velocity is the one quantity that crosses both messages**, and PX4 does
no conversion on it: `handle_message_hil_state_quaternion` copies
`rollspeed`/`pitchspeed`/`yawspeed` straight into
`vehicle_angular_velocity_groundtruth.xyz` (`SimulatorMavlink.cpp:563-565`). So the
FLU→FRD y/z negation is entirely ours, and it applies identically to `HIL_SENSOR`'s
`xgyro/ygyro/zgyro` and to `HIL_STATE_QUATERNION`'s body rates. Getting it right in
one and not the other gives a ground truth that disagrees with the IMU — which
reads as an estimator fault. Note the negation is the same as for a true vector
even though angular velocity is a pseudovector, because FLU→FRD is a *proper*
rotation (det `+1`).

**Both quaternions are body→world; do not conjugate.** MuJoCo's `qpos[3:7]`
rotates body-frame vectors into world. PX4's is the same direction:
`VehicleAttitude.msg` states `q` is the "Quaternion rotation from the FRD body
frame to the NED earth frame", and `simulator_mavlink` copies
`HIL_STATE_QUATERNION.attitude_quaternion` into it unchanged
(`SimulatorMavlink.cpp:572-581`). The conversion is a change of basis on both
ends, not an inversion:

```
q_px4 = q_ned_enu ⊗ q_mujoco ⊗ q_flu_frd⁻¹
  q_ned_enu = [0, √2/2, √2/2, 0]   # ENU→NED, 180° about (1,1,0)/√2
  q_flu_frd = [0, 1, 0, 0]         # FLU→FRD, 180° about x (self-inverse)
```

Body FLU→FRD is a 180° rotation about x: negate the y and z components of every
body-frame vector. World ENU→NED: `(x_n, y_e, z_d) = (y, x, -z)`; note this
matrix has determinant `+1`, so it is a proper rotation and composes cleanly with
the quaternion above.

**The geodetic origin is ours alone to choose.** In the mavlinksim path PX4 does
*not* read `PX4_HOME_LAT` / `PX4_HOME_LON` / `PX4_HOME_ALT` — those are consumed
only by the gazebo-classic plugins, `px4-rc.gzsim` and `px4-rc.sihsim`. PX4 takes
its local-tangent reference from the **first `HIL_STATE_QUATERNION` we send**:
`_global_local_proj_ref.initReference(lat, lon, timestamp)` runs once, on first
receipt, and `_global_local_alt0` is latched from the same message
(`SimulatorMavlink.cpp:604-609`). So neither the airframe file nor the `.post`
script can influence it; our config owns it outright. We should still accept
`PX4_HOME_*`-shaped env vars for familiarity, but that is our convention, not
PX4's. Match PX4's `MapProjection` maths so our local frame and PX4's agree.

#### Verified attitude mapping

Computed numerically from the formula above (MuJoCo 3.13.0 conventions, PX4 321
Euler order). **This table is the specification; Phase 2's tests must reproduce
it.**

| MuJoCo body rotation (FLU) | PX4 roll | PX4 pitch | PX4 yaw |
|---|---|---|---|
| identity | 0° | 0° | **+90°** |
| +30° about body x | **+30°** | 0° | +90° |
| −30° about body x | **−30°** | 0° | +90° |
| +30° about body y | 0° | **−30°** | +90° |
| +30° about body z | 0° | 0° | **+60°** |

Three consequences that are easy to get backwards:

1. **Roll keeps its sign.** FLU→FRD is a rotation *about x*, so x is the rotation
   axis and the similarity transform leaves it fixed: `C·Rx(θ)·C⁻¹ = Rx(θ)`. Only
   pitch and yaw flip sign.
2. **A MuJoCo identity attitude is PX4 yaw +90°, not 0°.** Under ENU, body x at
   identity points along world +x, which is **east**; east is heading 90° in NED.
   Zero PX4 yaw corresponds to the MuJoCo body pointing along world +y.
3. Yaw therefore carries both a sign flip and a +90° offset:
   `yaw_px4 = 90° − yaw_mujoco`.

Getting this wrong is the most likely cause of EKF2 divergence, ahead of any
noise-model detail. Phase 2 exists to isolate it.

### 3.7 `HIL_STATE_QUATERNION` encoding traps

Two unit issues at the encode boundary, both verified against
`common.xml` and PX4's decode:

- `vx` / `vy` / `vz` are **`int16`, cm/s**. Range ±327.67 m/s. Saturate rather
  than letting the cast wrap — a wrapped velocity is a sign flip in ground truth.
- `xacc` / `yacc` / `zacc` are documented in `common.xml` as **mG**, but PX4
  divides by `1000.f` and uses the result directly as **m/s²**
  (`SimulatorMavlink.cpp:618-621`). Encoding literal milli-g therefore produces a
  9.81× error. Send milli-m/s² and comment the discrepancy at the call site.
  These three are **also `int16`**, so at milli-m/s² the full scale is only
  **±32.767 m/s² (≈ ±3.3 g)** — saturate them exactly as for velocity. Irrelevant
  in hover, reachable in a phase-7 contact impact. This
  only feeds `vehicle_local_position_groundtruth.ax/ay/az`, which no module in this
  path consumes, so it is a logging-fidelity issue rather than a flight one — but
  §8 says scaling is correct at the boundary, so make it correct.

`lat` / `lon` are `degE7`, `alt` is `mm`. Attitude is `[w,x,y,z]`, copied into
`vehicle_attitude_groundtruth` unchanged.

### 3.8 Parameters `px4-rc.mavlinksim` sets for us

`px4-rc.mavlinksim:5-7` sets `EKF2_GPS_DELAY 10`, `EKF2_MULTI_IMU 3`,
`SENS_IMU_MODE 0`. The last two put ekf2 into **multi-EKF mode**
(`EKF2.cpp:2790-2793`: `sens_imu_mode == 0` ⇒ `multi_mode = true`). Two
consequences for bring-up:

- Strategy A publishes only IMU 0, so only instance 0 is ever allocated. The
  allocation loop is bounded (30 s of simulated time, or until arming) and exits
  cleanly (`EKF2.cpp:2856-2925`). This is expected, not a fault.
- **In multi mode ekf2 publishes `estimator_attitude` / `estimator_local_position`,
  not `vehicle_attitude` / `vehicle_local_position`** (`EKF2.cpp:58-61`);
  `ekf2_selector` republishes to the `vehicle_*` topics. When checking with
  `listener`, use the `vehicle_*` topics for the final estimate and the
  `estimator_*` ones to inspect a specific instance. Confusing the two wastes time
  in phases 1, 3 and 5.

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
│   ├── quad_x.xml              phase 3-5 airframe (FLU body frame, §3.6)
│   └── quad_x_arm.xml          phase 7 aerial manipulator
├── px4/
│   ├── 22001_mujoco_quad       PX4 airframe file (copied into PX4 tree)
│   └── 22001_mujoco_quad.post  starts the sensor_*_sim modules
├── scripts/
│   ├── run_sitl.sh             launches PX4 + simulator together
│   └── install_px4_files.sh    copies px4/ into the PX4 tree AND registers
│                               both files in the ROMFS CMakeLists (§5)
└── tests/
    ├── test_frames.py          the §3.6 attitude table, round-trips, geodetic
    ├── test_hil.py             message encode/decode against pymavlink
    └── test_vehicle.py         rotor model: hover thrust, torque balance
```

`viewer.py` stays optional and off by default: under lockstep the viewer's
event loop must not gate the physics loop, and headless CI must not need GL.

---

## 5. PX4-side configuration

Two files, installed into the PX4 tree by `scripts/install_px4_files.sh`. Keep
them in this repo so the PX4 checkout carries only a small, documented,
reversible patch.

**Copying the files is not sufficient.** ROMFS contents are enumerated
explicitly: `init.d-posix/airframes/CMakeLists.txt` calls `px4_add_romfs_files()`
with one literal filename per line, and `.post` files must be listed separately
(see `1010_gazebo-classic_iris_opt_flow` and its `.post` in that list). There is
no glob. An unregistered airframe file is not packaged into the built ROMFS and
PX4 fails at boot with `Error: no autostart file found`.

So `install_px4_files.sh` must both copy the files *and* insert two lines into
that CMakeLists. Make the insertion idempotent (skip if already present) and
provide a `--uninstall` path, because this is a real modification to the upstream
tree — the earlier goal of touching no upstream file is not achievable. The PX4
checkout therefore carries exactly one patched file, which `git diff` in the PX4
tree will show; keep it that way and do not let it grow.

**`px4/22001_mujoco_quad`** — airframe file, modelled on
`ROMFS/px4fmu_common/init.d-posix/airframes/10016_none_iris`. Because the
selector in `px4-rc.simulator` falls through to `px4-rc.mavlinksim` for anything
that is not sihsim / gz / jmavsim, a plain airframe id gets us the right
simulator with no script edit. Verified: with `PX4_SIMULATOR` unset,
`SYS_AUTOSTART = 22001` (non-zero, and not 10017), and `SIM_GZ_EN` at its default
of `0` (`gz_bridge/parameters.c:41`), every branch of the `if` chain fails and the
`else` is taken (`px4-rc.simulator:10-25`). Do **not** set `SIM_GZ_EN` in the
airframe file. Contents:

- `. ${R}etc/init.d/rc.mc_defaults`
- `CA_*` rotor geometry matching `models/quad_x.xml` exactly — same arm length,
  same rotor order, same spin directions. A mismatch here shows up as a slow
  yaw drift or a roll/pitch cross-coupling that is easy to misread as an EKF
  problem.
- `PWM_MAIN_FUNC1..4 = 101..104`
- `param set-default SENS_EN_BAROSIM 1`, `SENS_EN_MAGSIM 1`, `SENS_EN_GPSSIM 1`
  (declarative only — see §3.3; the `.post` starts the modules unconditionally)
- `IMU_INTEG_RATE` needs no entry: `px4-rc.simulator:5` already does
  `param set-default IMU_INTEG_RATE 250` for every posix simulator, overriding
  the firmware default of 200
  (`src/modules/sensors/vehicle_imu/imu_parameters.c:50`). Setting it again in the
  airframe file would be harmless but redundant; the value the simulator must
  match is 250 Hz.

**`px4/22001_mujoco_quad.post`** — sourced near the end of `rcS` (`rcS:363`,
`[ -e "$autostart_file".post ] && . "$autostart_file".post`), after
`px4-rc.simulator` has started `simulator_mavlink`:

```sh
sensor_baro_sim start
sensor_mag_sim start
sensor_gps_sim start
```

Ordering is not critical, but not for the reason it first appears: under lockstep
`simulator_mavlink start` **blocks `rcS` until our first `HIL_SENSOR` arrives**
(§3.2), so by the time `.post` runs the clock is already advancing. What makes the
ordering safe is that `sensor_*_sim` only need ground truth to be arriving *while
they run*, which it is. `.post` is also the documented hook and keeps the airframe
file itself unmodified.

Launch shape: `PX4_SYS_AUTOSTART=22001 ./build/px4_sitl_default/bin/px4`.
Airframe id 22001 sits inside the range the CMakeLists reserves with
`# [22000, 22999] Reserve for custom models`, which is the correct home for an
out-of-tree airframe. (An id like 4600 is also free today but sits next to the gz
block and risks colliding with a future upstream model.) Confirm 22001 is still
unused with a glob over `init.d-posix/airframes/` before committing.

---

## 6. Phases

Each phase ends with a runnable artifact and a check that can fail. Do not start
a phase before its predecessor's exit criteria pass — a frame bug found in phase
2 is a ten-minute fix, and the same bug found in phase 5 looks like an EKF
tuning problem.

### Phase 0 — Environment and skeleton

- `pyproject.toml` with pinned deps. `pymavlink` is **not yet installed** in the
  workspace venv; add it (pin the exact version).
- Install the two PX4 files from §5 *before* building: copy them and register
  both in `init.d-posix/airframes/CMakeLists.txt`. Skipping the registration is
  the most likely way to lose an hour in this phase.
- Build PX4 once: `make px4_sitl_default`. Confirm
  `build/px4_sitl_default/bin/px4` exists, that `22001_mujoco_quad` and
  `22001_mujoco_quad.post` are present under
  `build/px4_sitl_default/etc/init.d-posix/airframes/`, and that
  `sensor_baro_sim`/`sensor_mag_sim`/`sensor_gps_sim` appear in the built
  command list.

**Exit**: `python -m mujoco_px4_sitl --help` runs; PX4 boots with
`PX4_SYS_AUTOSTART=22001` and logs `Waiting for simulator to accept connection on
TCP port 4560`.

### Phase 1 — Minimal closed loop, physics stubbed out

Prove the protocol before trusting any dynamics. `sim.py` returns a hardcoded
level-hover state; no `mj_step` yet. `frames.py` is not needed — the constants
below are already expressed in PX4 frames.

- TCP server on 4560, accept one client. Tolerate and discard PX4's unsolicited
  `HEARTBEAT` and `COMMAND_LONG` (§3.1) without breaking MAVLink framing.
- Send `HIL_SENSOR` (`fields_updated = 0x3F`) at 250 Hz of *simulated* time with
  a synthetic monotonic clock, `id = 0`, accel `[0, 0, -9.81]` in FRD (level and
  stationary — see §3.6), gyro zero.
- Send `HIL_STATE_QUATERNION` on **every IMU frame (250 Hz)** — above PX4's 200 Hz
  request, which is not an integer divisor of the IMU rate (§3.1) — with
  **`attitude_quaternion = [1,0,0,0]`, i.e. identity in PX4's own FRD→NED sense**
  (heading north, level) — not the conversion of a MuJoCo identity, which would be
  yaw +90° (§3.6). Phase 1 has no `frames.py`, so state the PX4-frame constant
  directly and do not reason through MuJoCo at all here. Fixed home position.
  Populate the velocity fields too, not just position: `sensor_gps_sim` needs
  `lpos.vx/vy/vz`, which `simulator_mavlink` fills from this message's `vx/vy/vz`
  (`SimulatorMavlink.cpp:610-620`). Zero is a valid stationary value, but the
  fields must be present and the message must keep arriving. The lat/lon sent here
  becomes PX4's local-frame origin for the whole session (§3.6).
- Implement the §3.2 loop **complete**: the wall-clock pacer *and* the bounded-lead
  brake. The pacer is what governs PX4's own boot, which happens entirely inside
  this loop's first frames (§3.2) — omitting it looks fine in phase 1 and leaves
  the boot window unbounded. The brake is what keeps us from outrunning PX4 in
  phase 5. Neither half is optional, and neither is a "steady state only" concern.
- Log the first actuator message and assert `flags & 1` (lockstep) when it
  arrives — but treat its absence during the first seconds as normal, not as an
  error.
- Assert IMU timestamp monotonicity in code.
- Instrument the ratio of simulated time to wall-clock time and log it
  periodically. It should sit near `speed_factor`. A ratio that climbs without
  bound is the "we are outrunning PX4" failure from §3.2; a ratio that *collapses*
  toward zero while the brake's timeout logs means the brake is pacing the loop
  instead of the pacer — the ~100× silent slowdown described in §3.2. Both are
  invisible without this counter.

**Exit**: `commander status` shows no sensor timeouts; `listener sensor_baro`,
`listener sensor_mag`, `listener sensor_gps` all produce data — this is the
single most important check in the plan, because it is the runtime confirmation
of §3.3, which so far is established only by source reading. `ekf2 status`
reports the attitude filter running. Simulated time advances in step with PX4,
and `HIL_ACTUATOR_CONTROLS` is observed arriving at roughly the IMU rate once
Commander is up. The sim-to-wall time ratio settles near `speed_factor` and stays
there — neither growing nor collapsing (§7).

### Phase 2 — Frames, isolated and tested

`frames.py` plus `tests/test_frames.py`. Deliberately placed before any MuJoCo
integration: this is where the expensive bugs live, and every check here runs as
a plain unit test with no PX4 build and no simulator running.

- ENU↔NED, FLU↔FRD, quaternion conversion, geodetic projection.
- **Assert the §3.6 attitude table row by row.** It is the specification; do not
  re-derive the expected values while writing the test, or the test will encode
  whatever the implementation happens to do. In particular:
  - identity MuJoCo attitude ⇒ PX4 yaw **+90°**, roll 0, pitch 0;
  - `±30°` about body x ⇒ PX4 roll `±30°`, **same sign** (roll is the FLU→FRD
    rotation axis, so it is invariant — a test asserting `roll_px4 == -roll_muj`
    passes only for a *broken* conversion);
  - `+30°` about body y ⇒ PX4 pitch `−30°` (sign flips);
  - `+30°` about body z ⇒ PX4 yaw `+60°`, i.e. `yaw_px4 = 90° − yaw_muj`.
- Round-trip identity: `px4_to_mujoco(mujoco_to_px4(q)) == q` for a set of random
  quaternions. **Compare up to sign** — `q` and `−q` are the same rotation, and a
  naive `allclose(back, q)` fails spuriously on roughly half the inputs. Assert
  `allclose(back, q) or allclose(back, -q)`, or compare rotation matrices.
- A level stationary vehicle yields accelerometer `[0, 0, -9.81]` in FRD. MuJoCo
  reads `[0, 0, +9.81]` in its z-up site frame (verified empirically, §3.6); the
  FLU→FRD y/z negation flips the sign. Specific force at rest is the negative of
  the gravity vector — PX4's convention is accel z ≈ −9.8 when level, while the
  *gravity vector itself* in FRD is `+z`. Keep the two distinct; conflating them
  is exactly the bug this test catches.
- **Use pitch or yaw, not roll, to discriminate against a conjugated
  conversion.** Roll is invariant under FLU→FRD, so it carries no information
  about conjugation. A conjugate inserted "to match conventions" swaps roll and
  pitch wholesale — feeding `+30°` about body x yields `roll 0, pitch −30°,
  yaw −90°` instead of `roll +30°, pitch 0, yaw +90°`. Assert the full triple
  against the table, which catches this and any other basis error at once.
- Cross-check the geodetic projection against PX4's `MapProjection` behaviour
  using a handful of hardcoded reference points, with the origin taken from our
  own config (§3.6: PX4 adopts whatever origin our first `HIL_STATE_QUATERNION`
  carries, so the test owns both sides).
- Angular velocity: assert the y/z negation is applied, and that `HIL_SENSOR`'s
  gyro and `HIL_STATE_QUATERNION`'s body rates go through the *same* conversion
  (§3.6). A test that only covers one of the two misses the case where ground
  truth and IMU disagree.
- Test the §3.7 encode boundary here too: `int16` cm/s velocity saturation,
  `int16` milli-m/s² acceleration saturation at ≈ ±32.767 m/s², and the
  milli-m/s² scaling itself.

**Exit**: `pytest tests/test_frames.py` green, with no PX4 process involved.

### Phase 3 — Real MuJoCo state, open loop

Replace the stub with `models/quad_x.xml` and real `mj_step`, feeding everything
through the phase-2 `frames.py`. This is where phase 2's unit-level correctness
gets confirmed end to end against PX4. Actuator inputs still ignored; hold the
vehicle in place with a weld or by resetting state each step, so ground truth
moves in a known way.

- Verify `models/quad_x.xml` obeys **both** §3.6 model-authoring constraints — FLU
  body frame *and* ENU world frame (world +x = east) — before trusting any output.
  The second one has no local symptom; it shows up only as a 90° heading error
  later.
- IMU from MuJoCo `accelerometer` + `gyro` sensors on a body-fixed site.
- Simulation clock derived from `mj_data.time`, not wall clock.
- IMU at `IMU_INTEG_RATE` (250 Hz); physics `dt` an integer divisor of it
  (e.g. 1 kHz physics, IMU every 4th step). Record the ratio in `config.py`.

**Exit**: injecting a known attitude in MuJoCo produces the attitude predicted by
the §3.6 table in `vehicle_attitude_groundtruth` (compare via `listener`), for
roll, pitch, and yaw taken one at a time plus one combined 45°/45° case. Write the
expected numbers down before running it. **The values will not "match" naively** —
identity MuJoCo attitude reads as yaw +90°, and pitch/yaw are sign-flipped. That
is correct behaviour, not a bug to chase, and it is the single most common way to
waste a day in this phase.

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

## 7. Bring-up troubleshooting

**First, separate a hang from a divergence.** If the simulation stops advancing
rather than flying badly — PX4's log goes quiet, simulated time freezes, both
processes idle — that is not an estimator problem and nothing below applies. It
is almost always the §3.2 deadlock: we are waiting for `HIL_ACTUATOR_CONTROLS`
while PX4 waits for the `HIL_SENSOR` that would advance its clock. Confirm by
checking whether we stopped sending `HIL_SENSOR`, and fix the loop rather than
any parameter. With the §3.2 loop this can only happen if the brake's timeout is
missing or accidentally expressed in *simulated* time; that is the first thing to
check. If PX4 never finished booting at all — no `commander` output, `rcS` stopped
inside `px4-rc.simulator` — the cause is the same but earlier: PX4's boot itself
blocks on our first `HIL_SENSOR` (§3.2). A watchdog on *wall clock* should detect
and report both automatically.

**Second, check the sim-to-wall time ratio in both directions.** Two distinct loop
faults, neither of which any frame check below will find:

- *Ratio grows without bound* — we outran PX4. Under lockstep PX4 cannot get ahead
  of us, but nothing stops us getting ahead of PX4 (§3.2), and when we do, IMU FIFO
  samples are dropped before ekf2 consumes them. This looks exactly like estimator
  divergence. Cause: the wall-clock pacer is missing, or `MAX_LEAD_FRAMES` is far
  too large. Most likely during PX4's boot window, where the pacer is the only
  bound that exists.
- *Ratio collapses toward zero, brake timeout logging steadily* — the bounded-lead
  brake is pacing the loop instead of the pacer. Everything is correct but ~100×
  slow, and because expiry is non-fatal by design it presents as a working
  simulation rather than an error (§3.2). Cause: `MAX_LEAD_FRAMES` too small for
  PX4's actual actuator publish rate, or a brake that waits per-frame.

Fix the loop before reading further. Also note PX4's benign wall-clock
`poll timeout` error (§3.1) is not evidence of either.

### EKF2 divergence checklist

In observed likelihood order. Work top to bottom; do not start tuning EKF2
parameters until every item above the one you suspect has been excluded.

1. **Body frame sign error** — FLU vs FRD. Symptom: instant divergence on arming,
   or control that fights itself. Check that a level stationary vehicle sends
   accel `[0, 0, -9.81]` (FRD), and that `listener sensor_accel` in PX4 agrees.
2. **World frame swap** — ENU vs NED. Symptom: north/east transposed, or
   altitude inverted; position control drives the wrong way. Also check the model
   itself obeys world +x = east (§3.6): a model authored x-north produces a clean
   90° heading offset that looks like a conversion bug but is not.
3. **Non-monotonic or jittery IMU timestamps** — this *is* PX4's clock under
   lockstep. Symptom: erratic filter behaviour, lockstep stalls, log gaps.
   Assert monotonicity in code, not just in review.
4. **Wrong IMU rate** — must match `IMU_INTEG_RATE` (250 Hz).
5. **Quaternion convention** — `[w,x,y,z]` on both sides, and **body→world on
   both sides** (§3.6). MuJoCo's `freejoint` quaternion and PX4's
   `vehicle_attitude.q` (FRD→NED) are the same direction, so the conversion is a
   change of basis with **no conjugation**. Inserting a conjugate "to match
   conventions" is the classic error here; it presents as roll and pitch having
   swapped places, not as a plain sign flip. Diagnose by checking a pure body-x
   rotation: correct gives `roll ±30°, pitch 0`, conjugated gives
   `roll 0, pitch ∓30°`. Do not "fix" a correct conversion because yaw reads +90°
   at MuJoCo identity — that is expected (§3.6).
6. **`HIL_STATE_QUATERNION` rate too low or missing fields** — baro/mag/GPS all
   derive from it. PX4 asks for 200 Hz; we send it on every IMU frame instead
   (§3.1). If `sensor_gps` is stale, check that
   local-position ground truth carries velocity, since `sensor_gps_sim` reads
   `vx/vy/vz` from it. Also check the §3.7 encoding: an `int16` cm/s overflow
   turns a fast climb into a sign-flipped descent.
7. **Rotor geometry mismatch** between `CA_ROTOR*` and the MuJoCo model.
   Symptom: slow yaw drift, or roll/pitch coupling. Frequently misdiagnosed as
   an estimator fault.
8. Only now: EKF2 parameters.

Useful during bring-up: `listener <topic>`, `ekf2 status`, `commander status`,
`uorb top`. Remember that under multi-EKF (§3.8) the fused estimate is on
`vehicle_attitude` / `vehicle_local_position`, while `estimator_*` shows a single
instance. For a suspected sensor-synthesis problem, run the same airframe under
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
  boundary in `hil.py` — and there, follow **PX4's actual decode**, not the
  `common.xml` prose, where the two disagree (§3.7). Document each such case
  inline.
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
