# mujoco_px4_sitl

MuJoCo as the physics backend for PX4 SITL, over the MAVLink HIL interface.
Target application: aerial manipulation (multirotor + robotic arm).

The simulator is a standalone Python process. It owns physics and speaks PX4's
`simulator_mavlink` protocol; PX4 takes its clock from our IMU timestamps under
lockstep. Sensor strategy A: we supply IMU + ground truth, and PX4's own
`sensor_baro_sim` / `sensor_mag_sim` / `sensor_gps_sim` synthesize the rest.

No ROS 2 dependency here. See [ROS 2 integration](#ros-2-integration).

## Documents

- **`IMPLEMENTATION_PLAN.md`** — the PX4 v1.17 interface contract, the phase order,
  and the measured baselines. Read it before changing anything in `src/`.
- **`MODELING_CONVENTIONS.md`** — how to author a model: CAD → MJCF, the rotor
  parametrization, the naming the loader requires. Read it before adding a model.
- **`AGENTS.md`** — current state and what is unimplemented. Start here when
  picking the work back up.

## Setup

**The PX4 build requires the workspace venv on `PATH`.** PX4's build-time Python
dependencies (`kconfiglib`, `pyros-genmsg`) live in `../.venv`, which has
`include-system-site-packages = false`. PX4's cmake resolves its interpreter via
`PATH`, so without the venv it picks `/usr/bin/python3` and configure aborts in
`cmake/kconfig.cmake`. `make install_python_requirements` cannot bootstrap this:
it is a cmake target, so configure must already have succeeded.

```sh
source ../.venv/bin/activate          # before every PX4 build
./scripts/install_px4_files.sh        # airframe 22001 + its .post, into the PX4 tree
make -C ../PX4-Autopilot px4_sitl_default
```

`install_px4_files.sh` copies two files *and* registers them in
`ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt`. That registration is
mandatory — ROMFS contents are enumerated literally, with no glob, and an
unregistered airframe fails at boot with `no autostart file found`. It is
idempotent, and `--uninstall` reverses both steps. The PX4 checkout ends up
carrying exactly one patched file.

`--airframe PATH` installs a different airframe, including one from outside this
repo; airframes may be installed alongside each other. The `.post` always comes
from this repo and is copied under the target airframe's own name, because `rcS`
looks for `"$autostart_file".post`. See "Non-quad models" under Run.

## Run

```sh
./scripts/run_sitl.sh                 # PX4 + simulator, Ctrl-C stops both
./scripts/run_sitl.sh -g              # with the MuJoCo viewer
./scripts/run_sitl.sh -s 2            # 2x real time
./scripts/run_sitl.sh -- --stub-physics   # protocol only, no mj_step
```

Start order does not matter: we bind TCP 4560 first and PX4 retries `connect()`
until we accept. Under lockstep PX4's boot blocks on our first `HIL_SENSOR`, so
`rcS` will sit in `px4-rc.simulator` until the simulator is running.

Arm and fly from PX4's shell, or from another terminal against a daemonised
instance:

```sh
cd ../PX4-Autopilot/build/px4_sitl_default/rootfs
../bin/px4-commander takeoff
../bin/px4-listener vehicle_local_position
```

The simulator can also be launched directly, which is what a ROS 2 launch file
should do:

```sh
python -m mujoco_px4_sitl --instance 0 --model models/quad_x.xml
```

`--instance N` derives the HIL port as `4560+N` and the side-channel port as
`14650+N`, matching PX4's own convention.

### Non-quad models

`models/quad_x.xml` needs nothing further: `vehicle.py`'s placeholder motors
match its 4 rotors. Any other rotor count needs its parameters, via a conversion
sidecar:

```sh
python -m mujoco_px4_sitl -m /path/x8.xml --rotors /path/x8.conversion.yaml
MUJOCO_SITL_ROTORS=/path/x8.conversion.yaml ./scripts/run_sitl.sh -m /path/x8.xml
```

The environment variable is what lets `run_sitl.sh` fly a non-quad without a
flag of its own. There is deliberately **no default**: a model whose rotor count
does not match is rejected at load rather than flown with placeholder motors.

Only the rotor block and `omega_idle` are read, so the CAD export the sidecar
names need not be present — flying does not depend on the STLs.

A non-quad also needs its own PX4 airframe, since `CA_ROTOR*` must match the
model. `urdf_to_mjcf.py --emit-airframe PATH` generates one from the same
sidecar, and `install_px4_files.sh --airframe PATH` installs it from anywhere:

```sh
python scripts/urdf_to_mjcf.py x8.conversion.yaml --emit-airframe /path/22002_mujoco_x8
./scripts/install_px4_files.sh --airframe /path/22002_mujoco_x8
make -C ../PX4-Autopilot px4_sitl_default
PX4_SYS_AUTOSTART=22002 ./scripts/run_sitl.sh -m /path/x8.xml
```

The `.post` comes from this repo either way and is installed under the
airframe's own name, because `rcS` looks for `"$autostart_file".post`.

## Health check

Every status line reports the ratio of simulated to wall-clock time, plus the
counters that make the loop's own faults visible:

```
t_sim=    3.00s ratio=1.001 frames=751 act=726 brake=0 timeouts=0
```

`ratio` should sit near `--speed-factor`. A ratio that climbs without bound means
the simulator is outrunning PX4 and IMU FIFO samples are being dropped; a ratio
that collapses toward zero with `timeouts` climbing steadily means the brake is
pacing the loop instead of the wall clock. Both present as estimator divergence
and neither is visible without this counter. See `IMPLEMENTATION_PLAN.md` §3.2
and §7.

**`brake` and `timeouts` should both be 0 in healthy flight, and `ratio` alone
will not tell you otherwise.** `act` sitting a little below `frames` is normal —
PX4 publishes actuators at ~97 % of the IMU rate, since `HIL_ACTUATOR_CONTROLS`
is not an acknowledgement of `HIL_SENSOR` (§3.2) — and that shortfall on its own
must not cause any braking. If `brake` instead tracks `frames` divided by
`--max-lead-frames`, the loop is braking on a cadence rather than in response to
PX4, and at `--speed-factor 1.0` the pacer's sleep will absorb the cost so
`ratio` still reads 1.000. The line above is a real 3 s run against PX4 v1.17.0.

A model with an arm appends its own counters. `STALE` shows while the watchdog
has acted, and a negative `prop_clearance` means an arm capsule is inside a
propeller disc:

```
... brake=0 timeouts=0 arm_cmd=4846 age=0.004s stale_episodes=2 rejected=0 clamped=0 prop_clearance=-16.0mm intrusions=1
```

## Workspace manipulability

`scripts/manipulability.py` asks whether an arm can hold an arbitrary pose at an
arbitrary point. It needs no PX4, no flight and no physics step -- only forward
kinematics -- so it runs against any MJCF carrying an `ee` site.

```sh
python scripts/manipulability.py <model>.xml --task pose --slice y --view
python scripts/manipulability.py <model>.xml --task pose --freeze arm_joint3
```

The vehicle contributes **yaw only**: translations are ignored because a
free-flying base puts the workspace wherever it likes, and roll and pitch are not
steady states in hover. `--freeze` locks any joint, `base_yaw` included.

```
task            pose
movable         base_yaw, arm_joint0, ... , arm_joint4  (6 DoF)
reachable       N voxels
rank-deficient  N voxels (P% of reachable)
w (full rank)   median ...  max ...
```

The counts are left out on purpose: a voxel total is an arm's reach by another
name, and this model's geometry is private (`AGENTS.md` §4). Run the script to see
your own.

The headline is `rank-deficient`: voxels no configuration reaches at full task
rank, where the task cannot be executed however the arm approaches. Two traps the
script's docstring covers and the output repeats:

- **`w` does not compare across configurations.** It is an *m×m* determinant, so
  removing a joint drops a factor and usually makes the number *larger*. Compare
  rank deficiency, which is dimensionless.
- **A solid cloud shows only its shell.** Use `--slice` for a true cross-section;
  without it the interior is invisible behind the outermost voxels.

## Regression flight

Two scripts fly the Phase 5 profile and measure it. They are separate on purpose:
the flight reports only what it can see live, and hover accuracy is measured from
the log, where the datum correction can be done properly.

```sh
./scripts/run_sitl.sh &                          # wait for "Ready for takeoff!"
python scripts/fly_regression.py                 # arm, takeoff, 5 m square, land
python scripts/hover_error.py ../PX4-Autopilot/build/px4_sitl_default/rootfs/log/*/*.ulg
```

`hover_error.py` needs `pyulog` (`pip install -e '.[dev]'`). It prints the datum
offset it removed, then the settled hover error:

```
  datum offset    N -0.371  E -0.195  U -0.655 m -- removed
  steady-state hover, 151.3 s of samples:
    horizontal   max 0.115 m   median 0.056 m
    vertical     max 0.158 m   median 0.035 m
```

**Judge those against `sensor_gps_sim`'s own noise** — σ = 0.2 m horizontal,
0.5 m vertical — not against zero. `IMPLEMENTATION_PLAN.md` phase 5 holds the
baseline to compare against, and documents the two ways this measurement goes
wrong silently: `MAV_CMD_NAV_TAKEOFF` param7 is AMSL rather than relative, and
the estimate and ground-truth topics latch separate datums.

## Frames

Two model-authoring rules, both load-bearing for every conversion:

- **Body frame is FLU** — x forward, y left, z up, with the IMU site aligned.
- **World frame is ENU** — +x east, +y north, +z up.

A consequence that looks like a bug and is not: **a MuJoCo identity attitude is
PX4 yaw +90°**, and `yaw_px4 = 90° − yaw_mujoco`. Roll keeps its sign; pitch and
yaw flip. `tests/test_frames.py` asserts the whole table.

All conversion lives in `frames.py`. A rotation anywhere else is a bug.

## Side channel

UDP, our own JSON schema, versioned by the `v` field. PX4 never sees it. Send
`{"v":1,"type":"subscribe"}` to start receiving `ground_truth` messages, and
`{"v":1,"type":"arm_cmd","mode":"position","values":[...]}` to drive the
manipulator. Ground truth is published in PX4 frames (NED/FRD) so the two
channels can never disagree. `--no-sidechannel` disables it.

`scripts/sidechannel_example.py` is the reference client — standalone, importing
nothing from this package, which is what a ROS 2 bridge node should look like.

**If you poll slower than 50 Hz, drain the socket first.** Ground truth queues in
the receive buffer, so a client that reads once per second gets second-old state,
and one that reads after a long sleep gets state from the start of that sleep. The
example's `--latest` mode shows the pattern. Suspect this before believing any
disagreement between the side channel and PX4's own topics.

### Driving the arm

`values[i]` is `arm_joint{i}`'s absolute angle in radians, one per `arm_act*`
servo; a command of the wrong length, a mode other than `"position"`, or a
non-finite value is rejected with a warning. Values outside a joint's range are
clamped and counted. Optional fields:

| Field | Meaning |
|---|---|
| `seq` | yours, echoed back as `arm.cmd_seq`; not used for ordering |
| `state_time` | the `ground_truth.time` the command was computed from |

**Send commands as a stream.** A watchdog on simulated time treats a command
older than `--arm-timeout` (default 0.5 s; 0 turns it off) as a controller that
has stopped, and `--arm-on-timeout` says what the servos do then:

| Action | Servos | Models |
|---|---|---|
| `freeze` (default) | retarget to the pose reached | an arm driver whose watchdog stops the arm |
| `keep` | hold the last target | a servo bus with no watchdog |
| `limp` | torque off; the arm falls | a driver that cuts power on a fault |

The next command that is fresh on arrival resumes control, as a step: after a
stall, restart from the arm's reached pose, not the old plan. With `state_time`
the age runs from the state the command was computed from, which also catches a
controller that keeps sending on stale state, and a command older than the one
in force is dropped as out of order. Without it, age runs from arrival. Through
`run_sitl.sh`, pass the flags after `--`:
`./scripts/run_sitl.sh -m <model> -- --arm-timeout 0.2 --arm-on-timeout limp`.

For a model with an arm, `ground_truth` carries an `arm` block:

| Field | Meaning |
|---|---|
| `cmd_seq` | `seq` of the command in force, `null` before the first |
| `cmd_age` | simulated seconds since that command's `state_time` (or arrival) |
| `cmd_stale` | whether the watchdog has acted |
| `prop_clearance` | metres from the arm's collision capsules to the nearest propeller disc, on the pose reached; negative means inside. `null` if the model has no disc-shaped rotor sites |

**Propeller intrusion is reported, never blocked**: logged when it starts and
ends, and counted on the status line. Keeping clear is the controller's job.
Joint state is not on the side channel (`AGENTS.md` §3).

## ROS 2 integration

Keep it in a separate package. Launch this process with `ExecuteProcess`, and put
a thin bridge node there that talks to the side channel over UDP. PX4's own
uXRCE-DDS traffic goes directly between PX4 and ROS 2 and does not pass through
the simulator. Nothing in `src/` may import `rclpy` or `ament`.

## Tests

```sh
python -m pytest            # no PX4 build and no GL needed
```

No install step: `pyproject.toml` puts `src` on pytest's path, so a fresh
checkout runs the suite as-is. To use the simulator itself from outside this
directory, install it — `pip install -e ".[dev]"` — or set
`PYTHONPATH=src`, which is what `scripts/run_sitl.sh` does.

`test_frames.py` covers the attitude table and the geodetic projection,
`test_hil.py` the MAVLink encode boundary, `test_vehicle.py` the rotor model and
the model-authoring preconditions, `test_config.py` the flag combinations that
would otherwise fail silently, and `test_loop.py` runs the lockstep loop against a
fake PX4 — including the deadlock, runaway, back-pressure and shutdown cases that
a live smoke test cannot distinguish.

`test_rotorconfig.py` covers the sidecar parser, `test_urdf_to_mjcf.py` the
conversion and the generated PX4 airframe, and `test_manipulability.py` the
Jacobian metric — including a fixture whose rank deficiency is an analytic fact
rather than a measurement. All three build their own fixtures, so no CAD geometry
is needed — the X8's real numbers are private, and asserting them here would
publish them.
