# mujoco_px4_sitl — Implementation Plan

MuJoCo as the physics backend for PX4 SITL, connected over the MAVLink HIL
interface. Target application: aerial manipulation (multirotor + robotic arm).

This document records the verified PX4 v1.17 interface contract, the phase order
and exit criteria, and the measurements taken when each phase was met. Update it
when a decision changes; it is not a historical log, but the measured results it
records **are** kept — they are the regression baseline.

Its scope ends at the PX4 interface and the phase structure. Two siblings own the
rest, and win inside their own scope:

- **`MODELING_CONVENTIONS.md`** — where models come from (CAD → MJCF), the rotor
  parametrization, and the target platform. Supersedes Phase 4's thrust curve and
  Phase 7's model details.
- **`AGENTS.md`** — current state, what is unimplemented, and what to do next.
  Read it first when picking up the work.

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
MAX_LEAD_FRAMES = 32         # IMU frames we may have outstanding before blocking
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

**Invariant, and the reason this section states one at all.** The steps above are
a description; this is the property that must hold however they are factored:

> `frames_since_ack` equals the number of IMU frames sent since the last
> `HIL_ACTUATOR_CONTROLS` was *received*. **Every** path that receives one must
> zero it — the non-blocking drain and the brake alike.

Write the invariant into the implementation, not just the step sequence. A step
sequence does not survive refactoring: the reset above appears in two branches,
and pulling the shared "latch controls" part into one helper is the obvious
tidy-up, at which point the reset can travel with only one of the two callers.
That leaves a brake that fires every `MAX_LEAD_FRAMES` frames on a fixed cadence
regardless of how promptly PX4 replies, which is §7's "brake is pacing the loop"
fault — and it is invisible at `speed_factor = 1.0`, because the pacer's own sleep
absorbs the wasted wall clock and the ratio still reads 1.000. This is not
hypothetical; it is what the first implementation of this loop did, and the
measured table below was taken against it.

The check that discriminates it, and the one a test should assert: **against a PX4
that answers every frame, the brake must never fire at all.** If `brake` instead
tracks `frames / MAX_LEAD_FRAMES`, the invariant is broken somewhere.

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

The second bullet's premise did not survive measurement: once PX4 has answered
once, it answers every frame of a loop that waits for it, so the ratio never
slips. That is what the decided regime below rests on. The first and third
bullets still hold, and are why this loop stays as its fallback before PX4's
first answer and after a timeout.

`MAX_LEAD_FRAMES` and the timeout are both tunables, not contract. If physics is
slower than real time (contact-rich manipulation), the `sleep` never fires and the
bounded lead becomes the binding constraint, which is the correct degradation.

**Measured values, and why a generous timeout is wrong.** An earlier revision of
this plan said to set the timeout generously — hundreds of milliseconds — on the
grounds that its only job is to break a deadlock. That reasoning is incorrect, and
measuring it makes the mechanism plain. **While we are blocked in the brake, PX4's
clock is frozen, because only our `HIL_SENSOR` advances it. So the brake can only
ever be satisfied by an actuator message PX4 had already produced.** When that bet
fails, the entire timeout is wall clock burned for nothing, and the ratio collapses
exactly as §7 predicts. Measured with the phase-1 stub against a live PX4 (250 Hz,
`speed_factor = 1.0`):

| `MAX_LEAD_FRAMES` | brake timeout | sim/wall ratio |
|---|---|---|
| 8 | 500 ms | 0.40 |
| 32 | 500 ms | 0.86 |
| 32 | 50 ms | 1.000 |
| 64 | 100 ms | 1.000 |

**These four rows are observations and stand. The explanation below them was a
hypothesis, and it was wrong — read the two apart.** The distinction is the point:
everything in §3 up to here is either cited to a PX4 source line or measured, and a
causal story about *why* a number came out that way is neither. Written in the same
voice as the rest, it gets inherited as established fact, and then it decides where
the next person looks.

> **Superseded hypothesis.** *"Two independent causes, both present in the first
> row. The lead must be well above the actuator/sensor ratio's jitter — PX4
> publishes at ~86 % of the IMU rate while disarmed and ~98 % once warm, and its
> sender thread batches, so a lead of 8 brakes constantly."*
>
> The rates are real as rates of messages *received* while the loop runs ahead;
> PX4 itself computes an output for every frame (below, "What an answer proves").
> The conclusion drawn from them was not: the table was
> measured against an implementation that violated the invariant above, so the lead
> sensitivity it shows is that defect, not PX4's jitter. With the invariant held, a
> PX4 answering every frame does not brake at *any* lead — measured `brake=0` at
> leads of 8, 32 and 64, and `brake=0 timeouts=0` against a live PX4.
>
> This hypothesis survived because raising `MAX_LEAD_FRAMES` did improve the ratio,
> which is consistent with it — and also consistent with the defect. **A tunable
> that relieves a symptom shows interaction, not mechanism.** The check that would
> have separated them is one line and costs nothing: does braking happen when PX4
> answers every frame? Do not write a causal claim here without stating the
> observation that would refute it.

What still holds from that paragraph, on its own merits: the timeout must be
*short*, because a deadlock is diagnosed by *repeated* cheap timeouts, not by one
expensive one. Nothing is lost by retrying, since the pacer absorbs the slack and
the held setpoint is still correct.

**Defaults: `MAX_LEAD_FRAMES = 32`, brake timeout 50 ms.** The lead is now known to
be more conservative than needed, since the sensitivity that motivated 32 was the
defect; it is kept because a generous lead costs nothing and still bounds the
FIFO-drop hazard. Re-measure the whole table after any change to the IMU rate or to
PX4's publish behaviour; the ratio counter is what makes this visible, which is why
§7 leads with it.

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

#### What an answer proves, and what it does not

Read from source and measured with the Phase 2 probes (phase 7, "The PX4 legs"),
which ran a loop that sends frame k+1 only once PX4's answer to frame k is in:

- **PX4 computes an output for every frame.** After PX4's first answer, every
  frame got one: 447,811 of 447,811 frames over 40 runs, disarmed included, on
  the quad and the X8, idle and with every CPU core busy. The first answer came
  4–53 frames after PX4 connected. The shortfall the loop above sees (`act` below
  `frames`) is therefore the lead's: the sender wakes once, `orb_copy`s the newest
  output and sends one message (`SimulatorMavlink.cpp:170-176`, `:1044-1066`),
  however many were published while we ran ahead. That reading fits the source
  and the measurement; it was not isolated separately.
- **Under that loop, the stamp proves the frame.** `time_usec` is PX4's clock when
  it sends (`SimulatorMavlink.cpp:120`), which only our `HIL_SENSOR` moves. An
  answer stamped `t_k` was sent during frame k, and frame k was sent only after
  frame k−1's answer arrived, so it was computed from frame k. After the first
  answer, none carried another frame's stamp.
- **`px4_lockstep_wait_for_components()` covers the work queues, but does not
  prove the estimator finished.** A work queue registers as a lockstep component
  when work is queued and unregisters once drained (`WorkQueue.cpp:129-140`,
  `:195-197`), and the sender waits on the components before answering
  (`SimulatorMavlink.cpp:1064`). Yet with every core busy, EKF2's output fetched
  right after an answer was one frame old on every frame; idle, on none. An
  answer says nothing about that frame's estimate (§3.9). Why the wait misses it
  is not established. The semaphore holds at most one pending post
  (`lockstep_components.cpp`), which could leave it a post ahead, but that is a
  hypothesis: no observation has separated it from others.
- **Within a frame, PX4's threads race.** `WorkQueueManager.cpp:283-320` asks for
  `SCHED_FIFO` but never sets `PTHREAD_EXPLICIT_SCHED`, so on Linux the attribute
  is ignored: every PX4 thread runs `SCHED_OTHER` (policy 0 in
  `/proc/<pid>/task/*/stat` of a live PX4, `wq:rate_ctrl` included). The
  priorities that would run the rate loop ahead of the attitude loop do not exist
  in SITL. Measured consequence: an attitude setpoint reaches the rate loop one
  frame after a body-rate setpoint sent at the same time on 95–100 % of frames,
  and in the same frame on the rest (28 % with every core busy). Commander,
  navigator, the MAVLink main threads and the logger (started without `-p`, as
  `rc.logging` does) are plain threads sleeping on simulated time, not lockstep
  components; they run whenever their deadline
  falls and race the frame's chain.

#### Decided 2026-09-26: strict lockstep, the only regime — not yet implemented

The loop above is what `loop.py` does today. It is to be replaced; `AGENTS.md` §3
has the work list.

- **Once PX4 answers, frame k+1's `HIL_SENSOR` waits for the answer stamped
  `t_k`.** The lead is zero, and the brake has nothing to do in steady state.
- **IMU → actuator is exactly one frame.** The frame `[t_k, t_{k+1})` is driven
  by the answer to the `HIL_SENSOR` stamped `t_{k−1}`. One frame rather than zero:
  the loop above gave one frame on 64–90 % of frames, and the Phase 5 baselines
  and the derived attitude gains were flown on that.
- **The loop above stays, as the fallback**, pacer and brake with its invariant:
  before PX4's first answer, when PX4's chain is silent and blocking would
  deadlock, and for any frame whose answer does not arrive within a wall-clock
  timeout. Such a frame is counted as unproven, never silently, and is never
  fatal.
- **The pacer stays as a ceiling.** `speed_factor` still holds a run to real time
  for QGroundControl; an unpaced setting lets a batch run as fast as PX4 answers.
  Probed: about 12× on the quad and 7.7× on the X8 with 16 idle cores, about 2.5×
  with every core busy.
- **`px4_lag` becomes constant by construction**, and gives way on the status line
  to counts of answered and unproven frames.

The estimate and setpoint legs of an in-process controller build on this (§3.9).

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

**Strategy B — the simulator synthesizing baro, mag and GPS — was probed on
2026-09-26 and deferred until a use needs it** (`AGENTS.md` §3). What it would
buy is reproducibility and control of sensor noise:

- **Strategy A's noise cannot be repeated or tuned.** The three modules draw from
  one shared libc `rand()`, seeded once in `SensorBaroSim`'s constructor
  (`SensorBaroSim.cpp:44`), so the noise a sample gets depends on how many draws
  every module made before it — decided by boot, which runs on wall clock (§3.2).
  The amplitudes are hard-coded, not parameters: GPS 0.2 m horizontal and 0.5 m
  vertical white noise plus velocity noise (`SensorGpsSim.cpp:117-121`), baro
  1 Pa plus drift (`SensorBaroSim.cpp:156`), mag 0.02/0.02/0.03 G
  (`SensorMagSim.cpp:133`).
- **That noise dominates how far two identical runs drift apart.** Two runs under
  the decided regime (§3.2) held position 0.9–101 cm apart, depending on whether
  their draws happened to line up; with the same three sensors synthesized by the
  simulator from a seeded generator, 2–4 cm (phase 7, "The PX4 legs"). EKF2's
  states already differed 0.84 s into a run under strategy A.

What taking it over needs, all measured to work in the probe: set the `MAG` and
`BARO` bits of `HIL_SENSOR` (§3.4; `abs_pressure` is hPa, `SimulatorMavlink.cpp:307`),
send `HIL_GPS` (`SimulatorMavlink.cpp:408`), and stop the `.post` starting the
three modules — a configuration change. The mag field must agree with PX4's own
world magnetic model at the GPS position, or preflight checks object: evaluate
`geo_magnetic_tables.hpp` with PX4's bilinear lookup
(`geo_mag_declination.cpp:69-101`). The probe's sampling was mag 50 Hz, baro
25 Hz, GPS 10 Hz, PX4's amplitudes by default, one seeded stream per sensor. The
IMU noise gap (`AGENTS.md` §2) belongs in the same module when it lands. Phase 3
and Phase 5 would then need re-flying, since their baselines were measured under
strategy A.

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

### 3.9 The API link under lockstep

An in-process controller (`control.py`) talks to PX4 over its API/offboard MAVLink
link: UDP 14540 + instance, onboard mode, 4 MB/s (`px4-rc.mavlink`). What it can
rely on, read from source and measured with the Phase 2 probes (phase 7, "The PX4
legs"):

- **The receive thread runs on wall clock.** It polls the socket with a 10 ms
  wall-clock timeout (`mavlink_receiver.cpp:3180`), so it handles our datagrams
  while PX4's clock is frozen: in order, each message to completion before the
  next.
- **Handling is synchronous.** `SET_ATTITUDE_TARGET` publishes its setpoint and
  `offboard_control_mode` to uORB before returning (`mavlink_receiver.cpp:1616`;
  the setpoint itself only in the Offboard nav state), as do
  `SET_POSITION_TARGET_LOCAL_NED` (`:1034`) and the commands commander acts on
  (published as `vehicle_command`).
- **PING is answered from that same thread** (`:1728-1738`), through the link's
  immediate send path (`mavlink_main.cpp:754`). So a PING at the end of a
  datagram, once echoed, proves that everything before it is in uORB: a
  **barrier** that works while the clock is frozen. Measured: a body-rate setpoint
  sent behind it before frame k's `HIL_SENSOR` was used by frame k's rate loop on
  every frame, idle and with every core busy; without it, on every frame idle but
  only about a third under load. An echo takes 15–35 µs idle.
- **`COMMAND_ACK` is no barrier.** The receive thread publishes it to uORB
  (`mavlink_receiver.cpp:122-134`) and the main thread sends it on simulated time.
- **Two requests block the receive thread on simulated time:**
  `SET_MESSAGE_INTERVAL` (`:2245`), and `REQUEST_MESSAGE` for a stream the link
  does not have yet (`:762-790`). Both end in `configure_stream_threadsafe()`,
  which sleeps in `px4_usleep` until the main thread takes the request
  (`mavlink_main.cpp:1207-1230`). With the clock held, a barrier sent after one
  never echoes — measured: nothing in 2 s, then the echo 2 frames after the clock
  moved again. Send them only while frames advance without barriers, and wait for
  their `COMMAND_ACK` first.
- **Periodic streams race the estimator.** The main thread sleeps 1.5 ms of
  simulated time per iteration at this data rate (`mavlink_main.cpp:2301-2362`),
  so it runs once per frame and sends whatever is newest; whether the frame's
  estimate is out yet is a race.
- **`REQUEST_MESSAGE` sends from the receive thread, but only what is new.** It
  calls the stream's `send()` (`mavlink_receiver.cpp:762-790`), and `ODOMETRY`
  sends only on a new `vehicle_odometry` (`streams/ODOMETRY.hpp:64`). A stream
  given a very long interval (2·10⁹ µs) never sends on its own and becomes a pure
  on-request stream. Not −1: that deletes the stream (`mavlink_main.cpp:1176`),
  and the next request would re-create it through the blocking path above.
- **EKF2's output comes every other frame.** At the default `EKF2_PREDICT_US`
  its filter updates every 8 ms, and `ODOMETRY.time_usec` is the IMU sample time
  (`EKF2.cpp:1684`, `streams/ODOMETRY.hpp:66`); `ekf2_selector` republishes it as
  `vehicle_odometry` (§3.8). Its attitude comes every frame, but carries its
  publish time (`EKF2.cpp:1045`), so it cannot mark EKF2's progress.
- **Nothing proves a frame's estimate complete** (§3.2). Fetched right after the
  frame's answer, the frame's own output was there on every frame idle and on none
  with every core busy; the output one frame older was there on every frame of
  every run.
- **uXRCE-DDS shares none of this.** `uxrce_dds_client` handles inbound samples
  only after a `px4_poll` on uORB returns, with a 10 ms simulated-time timeout
  (`uxrce_dds_client.cpp:648-678`), so a sample sent while the clock is frozen
  waits for the next frame. The PING barrier does not carry over (§9).

**Decided 2026-09-26 — not yet implemented (`AGENTS.md` §3):**

- **Estimates are fetched, at a chosen age of at least one frame.** After every
  frame's answer, `REQUEST_MESSAGE(ODOMETRY)` behind a barrier, with the periodic
  stream silenced as above. The controller at `t_k` gets the newest output with
  sample time ≤ `t_k − d`, `d` ≥ 1 frame; its age is `d` or `d` + 1 frame, set by
  EKF2's 8 ms cadence. An output with sample time ≤ `t_k − d` that arrives only
  after `t_k` marks that sample late in the record, never silently. Age 0 would
  need PX4's thread states from `/proc` — Linux-only, and it slowed a loaded
  machine to real time — and was declined.
- **Setpoints and commands are sent on simulated time.** What the controller
  returns at `t_k` is sent behind a barrier before the `HIL_SENSOR` of
  `t_k + d_sp`, `d_sp` ≥ 1 frame, so PX4 processes that frame with it. A body-rate
  setpoint is then used by that frame's rate loop exactly, and moves the rotors
  one frame later (§3.2); an attitude setpoint one frame later again on most
  frames (§3.2). Arming, mode and takeoff commands take the same path, so a run
  can be armed at a simulated time; commander's 10 ms loop still picks the frame,
  within 1–2.

---

## 4. Repository layout

```
mujoco_px4_sitl/
├── IMPLEMENTATION_PLAN.md      this file: interface contract, phase design,
│                               measured baselines
├── MODELING_CONVENTIONS.md     model authoring: CAD -> MJCF, rotor
│                               parametrization, platform config
├── AGENTS.md                   handoff: current state, what is unimplemented,
│                               next steps, which document wins
├── README.md                   quickstart for users, kept short
├── pyproject.toml              deps: mujoco>=3.13, numpy, pymavlink
├── src/mujoco_px4_sitl/
│   ├── __init__.py
│   ├── main.py                 CLI entry point (script-launchable), and
│   │                           run() for an in-process caller
│   ├── __main__.py             python -m mujoco_px4_sitl
│   ├── config.py               dataclass config, CLI + file override
│   ├── transport.py            TCP server, MAVLink framing, reconnect
│   ├── hil.py                  HIL_SENSOR / HIL_STATE_QUATERNION encode,
│   │                           HIL_ACTUATOR_CONTROLS decode
│   ├── frames.py               MuJoCo <-> PX4 frame + geodetic conversion
│   ├── sim.py                  MjModel/MjData ownership, step, sensor read
│   ├── vehicle.py              rotor thrust/torque model, actuator mapping
│   ├── loop.py                 lockstep orchestration
│   ├── rotorconfig.py          conversion sidecar -> RotorModel (--rotors);
│   │                           owns RotorSpec, shared with urdf_to_mjcf.py
│   ├── sidechannel.py          UDP arm-command / ground-truth server
│   ├── arm.py                  arm_cmd -> data.ctrl with its watchdog;
│   │                           propeller clearance, reported not blocked
│   ├── control.py              in-process research controller: schedule,
│   │                           drops, observation, recorder
│   ├── px4link.py              PX4's API link: EKF2's ODOMETRY, as a
│   │                           companion computer reads it
│   └── viewer.py               optional mujoco.viewer, off by default
├── models/
│   ├── quad_x.xml              phase 3-5 airframe (FLU body frame, §3.6)
│   └── x8_arm.conversion.yaml.template
│                               sidecar schema, no geometry. The phase 7 model
│                               itself is generated into the private repo
│                               (MODELING_CONVENTIONS.md §6)
├── px4/
│   ├── 22001_mujoco_quad       PX4 airframe file (copied into PX4 tree)
│   ├── 22001_mujoco_quad.post  starts the sensor_*_sim modules; shared by
│   │                           every airframe, installed under its name
│   └── mujoco_x8.airframe.template
│                               non-quad airframe boilerplate; filled by
│                               urdf_to_mjcf.py --emit-airframe into the
│                               private repo, since CA_ROTOR*_P[XY] is geometry
├── scripts/
│   ├── run_sitl.sh             launches PX4 + simulator together
│   ├── install_px4_files.sh    copies an airframe into the PX4 tree AND
│   │                           registers it in the ROMFS CMakeLists (§5);
│   │                           --airframe takes one from outside the repo
│   ├── urdf_to_mjcf.py         CAD URDF + sidecar -> MJCF (+ airframe)
│   ├── fly_regression.py       flies the phase 5 profile over MAVLink;
│   │                           reports corner errors only, not hover accuracy
│   ├── hover_error.py          hover accuracy from a ulog, datum-corrected
│   ├── manipulability.py       per-voxel reachability point cloud, any joints
│   │                           frozen; kinematic only
│   └── sidechannel_example.py  reference side-channel client, imports nothing
│                               from this package
└── tests/
    ├── test_frames.py          the §3.6 attitude table, round-trips, geodetic
    ├── test_hil.py             message encode/decode against pymavlink
    ├── test_vehicle.py         rotor model: hover thrust, torque balance
    ├── test_config.py          flag combinations that would fail silently
    ├── test_rotorconfig.py     sidecar parsing; the silent-at-runtime errors
    ├── test_urdf_to_mjcf.py    conversion + generated airframe, own fixtures
    ├── test_arm.py             arm writer, watchdog, exact disc clearance
    ├── test_control.py         controller schedule, drops, command-age bound
    ├── test_manipulability.py  Jacobian metric on analytic fixtures
    └── test_loop.py            lockstep loop against a fake PX4
```

`viewer.py` stays optional and off by default: under lockstep the viewer's
event loop must not gate the physics loop, and headless CI must not need GL.

---

## 5. PX4-side configuration

Two files per airframe, installed into the PX4 tree by
`scripts/install_px4_files.sh`. Keep them outside the PX4 checkout so it carries
only a small, documented, reversible patch.

The quad's pair lives in this repo. A generated non-quad airframe lives with its
model — `--airframe PATH` installs from anywhere — but the `.post` is shared: it
is platform-independent, so one copy is installed under each airframe's own
basename, because `rcS` looks for `"$autostart_file".post`. Airframes coexist in
the tree; select one at boot with `PX4_SYS_AUTOSTART`.

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
block and risks colliding with a future upstream model.) 22001 is confirmed
unused across `ROMFS/` at v1.17.0 (`d6f12ad1c4`); re-check with a glob over
`init.d-posix/airframes/` after any PX4 version bump.

---

## 6. Phases

Each phase ends with a runnable artifact and a check that can fail. Do not start
a phase before its predecessor's exit criteria pass — a frame bug found in phase
2 is a ten-minute fix, and the same bug found in phase 5 looks like an EKF
tuning problem.

### Phase 0 — Environment and skeleton

**The PX4 build requires the workspace venv on `PATH`.** PX4's build-time Python
dependencies (`PX4-Autopilot/Tools/setup/requirements.txt`) are installed in
`../.venv`, not system-wide, and that venv has
`include-system-site-packages = false`. PX4's cmake resolves its interpreter with
`find_package(PythonInterp 3)`, which follows `PATH`: without the venv it picks
`/usr/bin/python3`, which has no `kconfiglib`, and configure aborts at
`cmake/kconfig.cmake:4`. `genmsg` is absent there too and would fail later, in
uORB generation (`Tools/msg/px_generate_uorb_topic_helper.py:46`). PX4's own
`make install_python_requirements` cannot bootstrap this: it is a cmake target,
so configure must already have succeeded.

```sh
source .venv/bin/activate            # before every PX4 build
make -C PX4-Autopilot px4_sitl_default
```

Repeat this in `README.md`. It is the one environment fact that neither
repository records.

Verified present: venv Python 3.12.3 with `mujoco 3.13.0`, `numpy 2.5.3`,
`pymavlink 2.4.49`, `pytest 9.1.1` and PX4's requirements (`kconfiglib 14.1.0`,
`pyros-genmsg 0.5.8`); gcc 13.3.0, cmake 3.28.3, ninja 1.11.1, ccache 4.9.1;
Gazebo Harmonic dev packages, which matter only because `default.px4board`
enables `GZ_BRIDGE` / `GZ_MSGS` / `GZ_PLUGINS` — dead weight for the mavlinksim
path, but they must still compile. `symforce` is deliberately absent:
`EKF2_SYMFORCE_GEN` is `OFF` unless `EKF2_MAGNETOMETER` or `EKF2_WIND` is off
(`src/modules/ekf2/CMakeLists.txt:34,45-47`), so SITL uses the checked-in
derivations. Do not "fix" its absence.

Already confirmed with a stock-airframe build, so do not re-derive it:
`build/px4_sitl_default/bin/px4` builds clean; `simulator_mavlink`,
`sensor_baro_sim`, `sensor_mag_sim`, `sensor_gps_sim` and `pwm_out_sim` are
registered in the generated command list
(`build/px4_sitl_default/platforms/posix/apps.cpp`) — §3.3's
`CONFIG_COMMON_SIMULATION` claim, confirmed at build level rather than by source
reading; `ENABLE_LOCKSTEP_SCHEDULER` is in the compile flags, so §3.2's clock
contract holds; booting `PX4_SYS_AUTOSTART=10016` stops at `Waiting for simulator
to accept connection on TCP port 4560` and blocks there, which is §3.2's boot
behaviour observed directly.

Remaining:

- `pyproject.toml` with pinned deps (`mujoco==3.13.0`, `pymavlink==2.4.49`).
- Install the two PX4 files from §5: copy them and register both in
  `init.d-posix/airframes/CMakeLists.txt`. Skipping the registration is the most
  likely way to lose an hour in this phase. `PX4-Autopilot/` is otherwise
  unmodified at v1.17.0 (`d6f12ad1c4`), so `git diff` there should show exactly
  that one file.
- Rebuild — only ROMFS is repackaged, so it is seconds, not minutes — and confirm
  `22001_mujoco_quad` and `22001_mujoco_quad.post` are present under
  `build/px4_sitl_default/etc/init.d-posix/airframes/`.

**Exit**: `python -m mujoco_px4_sitl --help` runs; PX4 boots with
`PX4_SYS_AUTOSTART=22001` — the id under test, not the stock airframe used above —
and logs `Waiting for simulator to accept connection on TCP port 4560`.

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
- **Log the brake counters too, and assert them.** The ratio alone is not
  sufficient: at `speed_factor = 1.0` the pacer's sleep absorbs a misfiring brake,
  so the ratio reads 1.000 while the brake fires on every `MAX_LEAD_FRAMES`th frame.
  `brake` and `timeouts` are what expose §3.2's invariant being broken, and §7's
  conformance list is the set of checks to write here rather than later.

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
expected numbers down before running it.

**Measured, all five cases exact to five decimals** (`--hold-pose
--inject-attitude ROLL,PITCH,YAW`, MuJoCo 321 Euler degrees in, PX4
`vehicle_attitude_groundtruth` out):

| MuJoCo attitude | PX4 `q` `[w,x,y,z]` | PX4 roll/pitch/yaw |
|---|---|---|
| identity | `0.70711, 0, 0, 0.70711` | 0°, 0°, **+90°** |
| roll +30° | `0.68301, 0.18301, 0.18301, 0.68301` | **+30°**, 0°, +90° |
| pitch +30° | `0.68301, 0.18301, −0.18301, 0.68301` | 0°, **−30°**, +90° |
| yaw +30° | `0.86603, 0, 0, 0.5` | 0°, 0°, **+60°** |
| roll+pitch 45° | `0.5, 0.5, 0, 0.70711` | +45°, −45°, +90° |

One trap in building the `--hold-pose` rig: an unsupported airborne body has
`qacc = g`, so its accelerometer reads **free fall**, which is wrong for something
being presented to PX4 as stationary. Zero `qacc` and recompute the
acceleration-stage sensors (`mj_rnePostConstraint` + `mj_sensorAcc`) to get
`Rᵀ·[0,0,g]` — what a real IMU on a stationary tilted vehicle reads. Skipping this
feeds EKF2 a zero accel vector and the attitude check drowns in estimator
complaints that have nothing to do with the frames. **The values will not "match" naively** —
identity MuJoCo attitude reads as yaw +90°, and pitch/yaw are sign-flipped. That
is correct behaviour, not a bug to chase, and it is the single most common way to
waste a day in this phase.

### Phase 4 — Rotor model and actuator mapping

**Quadrotor only, and complete.** The rotor parametrization has since moved on:
`MODELING_CONVENTIONS.md` §2.5 owns it, and supersedes the thrust curve below for
any new work. The exit criterion recorded here stands as the quad baseline.

- Map `HIL_ACTUATOR_CONTROLS.controls[i]` ∈ `[0,1]` to thrust and reaction torque
  per rotor, with first-order motor lag and a `CA_ROTOR*_KM`-consistent torque
  coefficient. The thrust curve itself is `MODELING_CONVENTIONS.md` §2.5's.
- Apply as MuJoCo forces on rotor sites; keep the model's rotor order identical
  to the `CA_ROTOR*` indices in the airframe file.
- Calibrate so that hover sits near mid-stick: total thrust at command 0.5
  should be close to vehicle weight. **Superseded.** §2.5's idle offset has since
  moved hover to command 0.450 and `MPC_THR_HOVER` to 0.45. The exit criterion
  below still holds; the 0.5 figure does not.

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

**Measured, arm → takeoff → hover → 5 m square in Offboard → land,
`models/quad_x.xml`**: passes. EKF2 reports `attitude: 1, local position: 1,
global position: 1` within ~20 s of boot, preflight is clean, the square's corner
errors are 0.14–0.66 m measured 20 s into each 5 m leg (settling, not converged),
and the vehicle lands disarmed with no failsafe at `ratio=1.000` throughout.

**Re-measured twice after the ω-based rotor model** (`MODELING_CONVENTIONS.md`
§2.5): once at `THR_MDL_FAC 0` / `MPC_THR_HOVER 0.45`, then again after §2.6's fix
at `THR_MDL_FAC 1.0` / `MPC_THR_HOVER 0.2025` / `MPC_THR_MIN 0.0144`. Both pass the
same procedure, `ratio=1.000 brake=0 timeouts=0` throughout, `Landing detected` →
`Disarmed by landing`, no failsafe.

| | Baseline (`k·u²`, hover 0.50) | ω-based, `fac 0` | ω-based, `fac 1` |
|---|---|---|---|
| altitude error | 0.13 m max, mostly < 0.07 | 0.158 max / 0.035 median | 0.153 max / 0.036 median |
| horizontal error | 0.17 m max | 0.115 max / 0.056 median | 0.121 max / 0.051 median |
| square corner error | 0.14–0.66 m | 0.02–0.06 m | 0.03–0.07 m |

**The two ω-based columns are indistinguishable, and that is the expected
result** — not evidence the fix did nothing. Position accuracy here is bounded by
`sensor_gps_sim`'s noise, and the 2.00× attitude gain the fix removes sits inside
PX4's default gain margin. A flight profile this gentle cannot separate them; the
gain ratio is asserted on the plant in `tests/test_vehicle.py` instead. What these
columns do establish is **no regression**, which is what a parameter change of
this kind needs from a flight.

Reproduce with `scripts/fly_regression.py` then `scripts/hover_error.py` on the
resulting ulog. Sampled over 151 s of settled hover (10 s discarded per still
stretch) from `vehicle_local_position` against
`vehicle_local_position_groundtruth`. **The
corner figures are not directly comparable** — both were read 20 s into each leg,
i.e. mid-settle, so they measure how far convergence had got, not steady-state
accuracy. Read them as "not worse".

**Neither flight would have caught the `fac = 0` defect, and that is worth
knowing.** It was found by reading `CA_ROTOR*_CT`'s definition against the
effectiveness matrix, not from flight data. The first flight was read as clearing
`fac = 0` on the strength of hover error alone; it did not. **A gentle profile
bounds how wrong a parameter can be before it shows, and that bound is generous** —
a 2× attitude gain flew through takeoff, a square and landing with no visible
symptom. Anything derived rather than measured needs an assertion somewhere it can
fail, not a flight that merely fails to contradict it.

**Three traps cost time here and will again.** All produce plausible-looking
numbers rather than errors:

- `MAV_CMD_NAV_TAKEOFF`'s param7 is **AMSL**, not relative. Home is 488 m, so
  passing `5.0` reads as 483 m below home; navigator rejects it with
  `Already higher than takeoff altitude`, the vehicle sits armed, and
  `Disarmed by auto preflight disarming` follows ~10 s later. `MIS_TAKEOFF_ALT`
  is the relative one.
- **Each topic latches its own datum.** In this run `ref_alt` differed by 0.655 m
  and `ref_lat` by 3.34e-6° (0.371 m north). Comparing raw `x`/`y`/`z` across the
  two topics folds that straight into the error: it reads as 0.53 m horizontal /
  0.77 m vertical with the *median nearly equal to the max*, which is the tell —
  GPS noise gives a median well below the max, a datum offset does not.
- **`MPC_THR_MIN` is a floor on the setpoint, so it lives in the allocator's
  domain too.** Left at its 0.12 default when `THR_MDL_FAC` went to 1, it clamps
  the command at `sqrt(0.12) = 0.346`, i.e. 0.66 of vehicle weight: the vehicle
  cannot descend and landing drifts. `0.12² = 0.0144` restores the same command
  floor. `MPC_THR_HOVER`, `MPC_THR_MIN` and `THR_MDL_FAC` are one setting.

The square was flown by streaming `SET_POSITION_TARGET_LOCAL_NED` at 20 Hz and
switching to Offboard with `MAV_CMD_DO_SET_MODE` (custom main mode 6) — the stream
must be live *before* the mode switch and must keep running. `MAV_CMD_DO_REPOSITION`
is not a shortcut worth taking: it returns `MAV_RESULT_ACCEPTED` and then does
nothing useful here.

Steady-state hover error, sampled over 48 s against the ground-truth topics:

| | max error |
|---|---|
| altitude (absolute geodetic, datum-free) | 0.13 m, mostly under 0.07 m |
| horizontal | 0.17 m |

**Judge this against what PX4's synthetic GPS can deliver, not against zero.**
`sensor_gps_sim` adds Gaussian noise of σ = 0.2 m horizontally and **σ = 0.5 m
vertically** (`SensorGpsSim.cpp:117-119`), and `EKF2_HGT_REF` defaults to `1`
(GNSS), so vertical accuracy is bounded by that 0.5 m — the filtered result above
is well inside it. Two consequences worth knowing before chasing a phantom:

- The vertical axis is inherently ~2.5× noisier than the horizontal here. A
  vertical error around 0.5 m is one sigma of PX4's own model, not a bridge fault.
- Sampled *during* the climb's settling transient rather than in steady state, the
  same setup reads ~0.44 m of altitude error. That is a transient, not a bias; let
  the hover settle before measuring, or it looks like a systematic offset.

Note also that `vehicle_local_position.ref_alt` and its ground-truth counterpart
differ slightly (EKF2 latches its own datum), so compare **absolute** geodetic
altitude between the two topics. Differencing the local `z` fields folds that
datum offset into the result.

If EKF2 misbehaves here, work down §7 in order before touching EKF2 parameters.

#### The X8

**Measured 2026-09-25 on the coaxial X8 + arm** (private model, airframe
`22002_mujoco_x8` generated from the same sidecar, stock PX4 attitude gains, arm
held at zero, since `arm_cmd` did not reach the physics yet). Same procedure as
the quad: passes, `ratio=1.000 brake=0 timeouts=0` throughout,
`Landing detected` → `Disarmed by landing`, no failsafe.

| | quad, ω-based `fac 1` | X8 |
|---|---|---|
| altitude error | 0.153 max / 0.036 median | 0.202 max / 0.047 median |
| horizontal error | 0.121 max / 0.051 median | 0.117 max / 0.050 median |
| square corner error | 0.03–0.07 m | 0.01–0.05 m |

Sampled over 164 s of settled hover, with the same GPS-noise floor as above.
Three things from the same ulog carry more weight than the table:

- `hover_thrust_estimate` settled on **0.2162, the generated `MPC_THR_HOVER`
  exactly**. PX4 found the same hover point the sidecar derived.
- The allocator achieved the full torque setpoint on every sample, with no
  actuator saturation. The decks share thrust equally, as they must while the
  coaxial `c_t` discount is unapplied. A steady pitch-rate integrator trims the
  CoM's offset from the rotor centre, since the arm is mounted forward.
- Yaw tracked to 0.2° through the whole profile, and roll/pitch stayed under 2°
  through the yaw steps below. So the index→ESC order is consistent with the
  model's sites.

**The regression profile does not exercise yaw, and yaw is where the X8
differs**, so attitude steps were flown as well. Yaw steps went through
`SET_POSITION_TARGET_LOCAL_NED` with the yaw field unmasked (`type_mask
0x08F8`), 15 s per step. Roll and pitch steps went through `SET_ATTITUDE_TARGET`,
±5° for 3 s each, with thrust at hover plus a small altitude PD. Everything was
read from ground truth. Overshoot is past the target; settling is the last exit
from a 5° band for yaw and from a 10 % band for roll and pitch.

| step | quad, stock | X8, stock | X8, `MC_YAWRATE_K 5` | X8, `K 5` + `MC_YAWRATE_MAX 45` |
|---|---|---|---|---|
| yaw 20° | 1–4° / 0.6 s | 10–14° / 5–8 s | 2–4° / 0.5 s | |
| yaw 90° | 4–5° / 1.1–2.3 s | 47–50° / 11 s | 42–46° / 4.7 s | 3–4° / 2.1 s |
| pitch 5° | | 16 % / 0.85 s | | |
| roll 5° | | ≤ 1.4 % / 0.4 s | | |

**Why:** PX4's rate controllers output a *normalized* torque, and the allocator
scales each axis so that ±1 is the vehicle's own authority on it
(`ControlAllocationPseudoInverse::updateControlAllocationMatrixScale`). The
physical loop gain is therefore the PX4 gain times (authority / inertia). Relative
to the quad, this X8 has about 1/20 of that in yaw, 1/13 in pitch and 1/5 in roll.
With stock gains the yaw-rate loop crosses over below the yaw attitude loop
(`MC_YAW_P 2.8`), which is the ordering that produces overshoot. The quad has
more than an order of magnitude of separation there. Roll and pitch sit in
between, matching the table.

The 90° row has a second cause that `K` alone does not fix. At hover a motor
reaches zero within the first sample, and yaw acceleration tops out near
80°/s². The stock 200°/s rate limit then commands more yaw rate than the
vehicle can brake. Capping the rate removes it. Both yaw fixes were set at
runtime only, to confirm the diagnosis.

**The derived gains**, which `--emit-airframe` now writes (the airframe template
has the rule). For this X8 they are `MC_ROLLRATE_K 1.00`, `MC_PITCHRATE_K 2.18`,
`MC_YAWRATE_K 5.00` and `MC_YAWRATE_MAX 29.6`. Measured 2026-09-25 with the same
step procedures:

| step | X8, stock | X8, derived |
|---|---|---|
| yaw 20° | 10–14° / 5–8 s | 0.2–2.0° / 0.6–0.7 s |
| yaw 90° | 47–50° / 11 s | 0.0–0.6° / 2.9–3.1 s |
| pitch 5° | 16 % / 0.85 s | 1.2–2.1 % / 0.4 s |
| roll 5° | ≤ 1.4 % / 0.4 s | ≤ 1.5 % / 0.45 s |

The 90° step now takes 3 s because it is rate-limited, which is the point: the
vehicle turns at a rate it can brake from. The Phase 5 profile passes with them:
hover 0.160 m horizontal / 0.206 m vertical max (0.054 / 0.055 median),
corners 0.03–0.10 m, allocator torque achieved on 99.1 % of samples,
`brake=0 timeouts=0`, `Disarmed by landing`.

**The first version of the rule matched the quad's physical loop gain instead,
and it flew worse.** That gave roll K 4.78, pitch 5 and yaw 5. It passed the
Phase 5 profile, and hover rate noise was lower than on stock gains. But after
the first yaw step drove a motor to zero, roll entered a limit cycle, with rate
RMS around 11°/s against 0.6–0.9 before it. The torque setpoint swung ±0.3–0.46
and motors bounced between 0 and 0.55. It lasted more than a minute and
restarted on the next steps. The frequency is tens of Hz, too high for the logs
to pin down; the angle amplitude was only about 0.1°. Pitch at K 5 did not
oscillate. At the same loop gain this X8 has about 4.5× less acceleration
headroom than the quad in roll at hover, so saturation arrives on much
smaller signals. Neither a hover nor a gentle square can show
this, which is why the step tests ran against every candidate.

**A trap in re-running any of this:** after `Disarmed by landing` PX4 stays in
Land, and refuses to arm from there with `Arming denied: Resolve system health
failures first`. The message names neither the mode nor a sensor, and
`commander check` lists only unrelated failsafe flags. Switching to Hold first
arms normally, and `fly_regression.py` now does that.

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

**Met.** `scripts/run_sitl.sh` brings up both processes and stops both on Ctrl-C;
`scripts/sidechannel_example.py` reads ground truth at 50 Hz under the *system*
Python, with no `PYTHONPATH` and no imports from this repository.

Two things found while getting there, both worth keeping:

- **A consumer slower than the publish rate reads stale state.** 50 Hz into a
  socket buffer means a client that polls after an 18 s wait receives a datagram
  from the *start* of that wait, not the newest one. This cost an hour of chasing
  a phantom "the vehicle will not translate" bug: PX4 was flying the square
  correctly and the harness was reporting the pre-takeoff position. Any consumer
  that samples on its own timer must drain to the newest message; the example
  client's `--latest` mode is the reference pattern. Consider it before believing
  any disagreement between the side channel and PX4's own topics.
- `run_sitl.sh` must pass PX4 `-d` when stdout is not a TTY, or the `pxh` shell
  redraws its prompt into the log without bound.

### Phase 7 — Aerial manipulator

**`MODELING_CONVENTIONS.md` owns the model, the platform and the rotor
parametrization for this phase.** Target is a coaxial X8 octorotor plus serial arm,
authored in SolidWorks. This section keeps only the phase's intent and exit
criterion.

- `x8_arm.xml`: multirotor plus serial arm, actuated joints with realistic
  limits, torque limits, and mass. Generated by the conversion pipeline of
  `MODELING_CONVENTIONS.md` §6, not hand-authored, and generated **into the
  private repo** — it and its PX4 airframe are geometry (§6). *Done, and it
  loads under PX4's HIL with its measured allocation, and flies phase 5's
  profile with the quad's numbers (phase 5, "The X8"), on attitude gains the
  airframe generator derives for it.*
- Arm joints driven from the side channel, position-controlled
  (`MODELING_CONVENTIONS.md` §3.4). *Done: `arm.py` writes `data.ctrl`, with a
  command watchdog on simulated time and a propeller clearance check that
  reports and never blocks (`README.md`, "Driving the arm").*
- Expected physics coupling: the arm moves the composite CoM and adds reaction
  torques on the base. PX4 will see this as a disturbance, which is the point.
- Investigate whether the arm needs its own control rate distinct from the
  physics rate. *It has one: an in-process controller runs at its own sample
  period, a whole number of IMU frames (`control.py`).*

**Exit**: arm motion during hover produces a measurable, bounded attitude
disturbance that PX4 rejects without losing position lock. Ground truth for
both base and arm is available externally. *Met: the disturbance is measured
below, and base and arm ground truth reach an in-process controller's recorder.
The side channel still carries no arm joint state, by choice (`AGENTS.md` §3).*

#### Arm motion in hover

**Measured 2026-09-25 on the coaxial X8 + arm**, airframe `22002_mujoco_x8` with
its derived attitude gains, Offboard position hold at 3 m. Arm commands streamed
over the side channel at 50 Hz with `state_time` echoed, cosine ramps over 3 s.
Maxima against ground truth; position is the deviation from the mean of the
still hover before the first move, which needs no datum correction.
`ratio=1.000 brake=0 timeouts=0` throughout, no failsafe, landed and disarmed.

| Segment | Roll | Pitch | Yaw | Horizontal | Vertical |
|---|---|---|---|---|---|
| still hover, arm at zero | 1.15° | 0.59° | 0.11° | 0.173 m | 0.128 m |
| shoulder 0 → 90°, hold | 0.65° | 1.61° | 0.08° | 0.137 m | 0.099 m |
| arm yaw 0 → +90°, arm out | 0.73° | 1.01° | 0.23° | 0.112 m | 0.076 m |
| arm yaw +90° → −90° | 1.15° | 0.68° | 0.35° | 0.139 m | 0.129 m |
| back to zero | 0.63° | 1.88° | 0.40° | 0.146 m | 0.134 m |
| ramp, 3 s silence, resume | 1.00° | **7.83°** | 0.43° | 0.156 m | 0.080 m |
| into a lower-deck disc, hold (second flight) | 1.26° | 0.93° | 0.46° | 0.246 m | 0.252 m |

What is worth keeping from it:

- **Planned moves are rejected without losing position lock**, within 2° of
  attitude. The shoulder and arm-yaw moves stay inside the first flight's
  still-hover spread; folding the arm back over a disc moves the CoM furthest
  and cost 0.25 m, against 0.05–0.07 m of still hover in that flight. This half
  of the exit criterion holds at this speed.
- **The largest disturbance came from recovering from a stall.** The watchdog
  froze the arm 0.50 s into the silence, a quarter of the way up a shoulder
  ramp; the first command after it asked for the ramp's end, and the servo took
  it as a step. After a stall a controller has to restart from the reached pose.
- **The reached pose decides an intrusion, not the commanded one.** With the
  provisional gains gravity pulls shoulder and elbow about 4° off target: one
  pose commanded inside a disc ended 6.6 mm clear, another commanded just
  inside sagged until the capsule's axis crossed it. That one was reported at
  its start and its end, 7.3 s later, and nothing stopped it.
- **Command age is measurable out of process, not reproducible.** `arm.cmd_age`
  read 20 ms, one publish period, on every sample at 1×: wall-clock latency
  (`AGENTS.md` §3).

#### The in-process controller

**Measured 2026-09-25 on the coaxial X8 + arm**, same airframe, two flights of
160 s and 130 s. Controller in process (`control.py`): period 20 ms, delay 8 ms,
`CappedDrops(p=0.2, max_consecutive=2, seed=1)`, so the arm command age bound is
3 × 20 + 8 = 68 ms. Takeoff to 5 m and Hold from a GCS script; the controller
swept `arm_joint0` ±0.5 rad at 0.2 Hz for 30 s, starting 10 s after EKF2's
altitude estimate first read above 3 m. Both flights `ratio=1.000 brake=0
timeouts=0`, no failsafe, landed and disarmed.

| Quantity | Flight 1 | Flight 2 |
|---|---|---|
| samples, all on the 20 ms grid | 8000 | 6500 |
| dropped, longest run | 1526, 2 | 1245, 2 |
| `cmd_age` at sample instants, max | 60 ms | 60 ms |
| PX4 IMU → actuator lag (`px4_lag`) | 1 frame on 64.5 % of frames, max 2 | 1 frame on 63.0 %, max 2 |
| EKF2 estimate age at sample instants, median | 8 ms | 4 ms |

Flight 1 in more detail: estimate age p99 12 ms, max 40 ms; an estimate reached
the host a median 4 ms, at most 12 ms, after EKF2's sample. PX4 accepted
`ODOMETRY` at 250 Hz, and 7959 of the 8000 samples saw a new estimate. Airborne
(67 s), the estimate against ground truth in EKF2's frame: 0.057 m median /
0.138 m max horizontal, 0.043 / 0.176 m vertical. The sweep cost at most 2.63°
of tilt, and the arm stayed 141 mm clear of every disc.

What is worth keeping from it:

- **The arm leg is deterministic; the PX4 legs are not.** Across the two
  flights the sample instants and the drop pattern are identical, and the arm's
  joint trajectory is bit-identical until arming, which a wall-clocked GCS
  script timed differently. The median estimate age halved between the flights
  under the same schedule: that is the leg wall clock decides (`AGENTS.md` §3).
- **Without the datum rebase the estimate reads 0.42 m / 0.99 m off**, horizontal
  and vertical: EKF2's origin against the simulator's home, the trap
  `scripts/hover_error.py` describes. `Truth.pos_ned_ekf` removes it.
- **The frame-sampled `cmd_age` peak is one frame below the bound**, 64 ms here,
  and sample instants see at most 60 ms. `tests/test_control.py` asserts the peak
  is reached, so a loose bound would show.
- **EKF2's `reset_counter` stepped four times** in flight 1, three during
  alignment in the first 3 s and once at takeoff (t = 12.4 s). A controller that
  differentiates the estimate has to watch it.

#### The PX4 legs (Phase 2 probes)

**Measured 2026-09-26**, to decide phase 2 of the controller topology
(`AGENTS.md` §3); the facts drawn from them are in §3.2, §3.3 and §3.9. A probe
outside this repository (local, unversioned) started a fresh PX4 per run in a
throw-away rootfs, ran a lockstep loop built from this package's physics and HIL
transport, owned PX4's API link, and flew a mission on simulated time: Offboard
position hold at 3 m (setpoints from 18 s, armed at 20 s), a 3 s body-rate window
at 30 s with a host attitude loop on the fetched estimate, and a 3 s attitude
window at 38 s, both with a seeded ±0.015 thrust PRBS per frame. Which frame's
thrust reached the rotors is read from Σu² of the answer, linear in collective
thrust at `THR_MDL_FAC 1` and blind to torque: the right lag leaves a worst
residual of 0.9–1.5 % of one PRBS flip over a window, a wrong one a full flip.
Quad unless stated. "Strict" is the regime §3.2 decides, except that the probe
applied each answer to its own frame; the decided one-frame delay does not change
which setpoint an answer carries. "Loaded" is all 16 cores kept busy.

| Quantity | Measured |
|---|---|
| answers stamped with their own frame, after PX4's first | 447,811 of 447,811 frames over 40 runs: disarmed and armed, quad and X8, idle and loaded |
| PX4's first answer | 4–53 frames after it connected |
| wait for the answer | median 0.05 ms, p99 0.11 ms idle; median 0.5–1.1 ms, p99 about 4 ms loaded |
| body-rate setpoint behind the PING barrier | used by its frame on every frame: idle and loaded, quad and X8 |
| the same without the barrier | every frame idle; about a third loaded, the rest one frame late |
| attitude setpoint behind the barrier | one frame after a body-rate one on 95–100 % of frames, the same frame on the rest; 28 % the same frame loaded |
| EKF2 output of the frame itself, fetched after the answer | every frame idle; no frame loaded (5,498 of 5,498 one frame old) |
| EKF2 output one frame older | every frame of every run |
| `SET_MESSAGE_INTERVAL`, then a barrier, clock held | no echo in 2 s; the echo came 2 frames after the clock moved |
| bounded-lead loop, for comparison | body-rate setpoint one frame late on about 90 % of frames, the same frame on 8 %, two late on 2 %; attitude two late on 81–84 %, else one; `px4_lag` 1 frame on 90–92 %, 0 on the rest |

How far two runs of one schedule drift apart, position against ground truth,
fresh PX4 each:

| Loop | Baro, mag, GPS | Pairs | First difference | Hover 22–30 s, max | Whole flight, max |
|---|---|---|---|---|---|
| bounded lead (today) | PX4's `sensor_*_sim` | 1 | the arming frame | 61 cm | 83 cm |
| strict, and variants | PX4's `sensor_*_sim` | 10 | the arming frame | 0.9–101 cm | 1.4–124 cm |
| bounded lead | simulator, seeded | 1 | the arming frame | 19 cm | 21 cm |
| strict | simulator, seeded | 1 quad, 1 X8 | the arming frame | 3.6 cm, X8 0.9 cm | 3.6 cm, X8 2.1 cm |

The strict variants: ground truth sent before the IMU sample; PX4 pinned to one
core; strict from PX4's first frame, a frame with no answer ending when no thread
of PX4 or of the processes it starts is runnable (`rcS` runs as `/bin/sh` plus
`px4-*` clients); and a two-phase frame, in which a `HIL_SENSOR` with no fields
advances the clock, the loop waits until PX4 is idle, and the IMU sample follows
at the same timestamp.

Speed, strict and unpaced, simulated seconds per wall second: quad 11.4–12.7 and
X8 7.7–7.9 idle; quad 2.3–2.6 loaded. The `/proc` checks cost the variants half
of that or more.

What is worth keeping from it:

- **The legs as §3.2 and §3.9 decide them are what these numbers support.** The
  barrier looks unnecessary on an idle machine and is not: load is exactly when a
  setpoint misses its frame. Likewise, an estimate fetched after the answer looks
  complete until the machine is busy.
- **Runs match bit for bit until arming only because nothing moves before it.**
  The first armed output already differs.
- **PX4's sensor noise dominates the drift** (§3.3): in one pair's ulogs EKF2's
  states differ from 0.84 s, and seeded simulator-side sensors cut the spread by
  one to two orders of magnitude.
- **Arming lands 1–2 frames apart** even with the command in uORB at the same
  simulated instant: commander's 10 ms loop keeps a phase set during boot.
- **A deterministic boot was tried and declined.** With the process-tree idle
  check and the two-phase frame, one pair got its first answer (frame 4), first
  estimate (frame 78) and EKF2's cadence identical, the next pair its first
  estimate two frames apart; the estimate's content differed in both. With seeded
  sensors added, hover still differed by 0.6 cm, at half the speed. Bit-identical
  runs would need PX4 changes (§1).

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
  brake is pacing the loop instead of the pacer. Everything is correct but slow, and
  because expiry is non-fatal by design it presents as a working simulation rather
  than an error (§3.2). **Suspect the loop code before the tunables**: the first
  implementation of this loop produced exactly this signature with well-chosen
  values, because the lead counter was not reset on the drain path (§3.2's
  invariant). Only then consider `MAX_LEAD_FRAMES` too small for PX4's actual
  actuator publish rate, or the brake timeout set too long — under lockstep a
  blocked brake freezes PX4's clock, so a long timeout is wall clock burned for
  nothing. §3.2 has the measured table.

Fix the loop before reading further. Also note PX4's benign wall-clock
`poll timeout` error (§3.1) is not evidence of either.

**Third — and before the estimator checklist — verify the loop implements §3.2,
rather than assuming it.** Everything below diagnoses a *misbehaving system*; this
step asks a different question, which nothing else here asks: does the code do what
§3.2 says? Three of the defects found in review were in this gap. They are cheap to
exclude and each has a mechanical check:

- **The lead counter is reset on every receive path**, drain and brake alike (§3.2's
  invariant). Check: with a PX4 answering every frame, `brake` must be 0. If it
  tracks `frames / MAX_LEAD_FRAMES`, one path is missing its reset — and note the
  ratio still reads 1.000 at `speed_factor = 1.0`, so this counter is the only
  witness.
- **The brake's timeout is wall clock**, never simulated time (§3.2). A timeout in
  simulated time never fires, which is the deadlock above.
- **Socket back-pressure is not mistaken for a disconnect.** A full send buffer is a
  PX4 that stopped reading, not a PX4 that went away; on a non-blocking socket it
  arrives as `BlockingIOError`, which is an `OSError`, so a broad `except OSError`
  around the send will tear down a live link. A partial write abandoned mid-frame
  desynchronises PX4's parser silently, which then presents as §3.1's framing
  problem with no apparent cause. Check: a peer that stops reading must stall the
  send, not drop the connection.
- **`stop()` is honoured in every state the loop can be in**, including waiting for
  PX4 to connect — PX4 may never connect at all, a wrong airframe id is enough. A
  wait that ignores SIGINT/SIGTERM hangs `run_sitl.sh`'s cleanup, which signals and
  then waits.
- **PX4 cannot finish exiting once simulated time stops.** Its `shutdown` runs on
  the work queue, which under lockstep runs on our clock, so after the simulator
  stops PX4 prints "PX4 Exiting..." and stays; one still blocked in boot never
  exits either. An unbounded wait on it hangs the launcher and orphans PX4, which
  then holds its instance against the next run ("PX4 server already running").
  `run_sitl.sh` stops PX4 first, while the simulator still runs, and escalates to
  SIGKILL after 3 s.

These are conformance checks, not tuning. Assert them in `tests/test_loop.py`
against a fake PX4 that is *uncooperative* — one that never replies, never reads,
or never connects. A cooperative fake exercises none of them, and a successful
Phase 5 flight does not either: the loop is deliberately fault-tolerant, so a
mechanism can misfire continuously while the vehicle still flies.

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
  in-process `rclpy` bridge stays possible without a rebuild). The same venv also
  carries PX4's build-time dependencies and must be active for any PX4 build —
  see Phase 0.
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

Resolve when the phase that needs them arrives, not before. Questions about the
model, the platform or the rotor parametrization belong in
`MODELING_CONVENTIONS.md` §8, not here.

- Physics rate for contact-rich manipulation: 1 kHz may not be enough; measure
  before deciding, and note that raising it costs wall-clock time only, not
  correctness. `MODELING_CONVENTIONS.md` §5 has the headroom measurement this
  needs — 74k steps/s on `quad_x.xml`, 296 steps per IMU frame at real time.
- Whether the aerial manipulator eventually needs a custom `CA_*` allocation or
  benefits from PX4's existing disturbance rejection alone. Still open, and the
  move to X8 does not settle it.
- How to put a uXRCE-DDS thrust/torque setpoint in place before a frame. The
  client reads inbound samples only when its simulated-time `px4_poll` returns
  (§3.9), so the MAVLink PING barrier does not apply, and in thrust/torque mode
  the allocator runs only on those samples, so a late one also delays that
  frame's answer. Needs evidence that a sample reached the client before the
  frame; design it once an agent and `px4_msgs` are installed.
