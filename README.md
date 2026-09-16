# mujoco_px4_sitl

MuJoCo as the physics backend for PX4 SITL, over the MAVLink HIL interface.
Target application: aerial manipulation (multirotor + robotic arm).

The simulator is a standalone Python process. It owns physics and speaks PX4's
`simulator_mavlink` protocol; PX4 takes its clock from our IMU timestamps under
lockstep. Sensor strategy A: we supply IMU + ground truth, and PX4's own
`sensor_baro_sim` / `sensor_mag_sim` / `sensor_gps_sim` synthesize the rest.

No ROS 2 dependency here. See [ROS 2 integration](#ros-2-integration).

`IMPLEMENTATION_PLAN.md` is the reference for the PX4 v1.17 interface contract
and the phase order. Read it before changing anything in `src/`.

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

## ROS 2 integration

Keep it in a separate package. Launch this process with `ExecuteProcess`, and put
a thin bridge node there that talks to the side channel over UDP. PX4's own
uXRCE-DDS traffic goes directly between PX4 and ROS 2 and does not pass through
the simulator. Nothing in `src/` may import `rclpy` or `ament`.

## Tests

```sh
python -m pytest            # 84 tests, no PX4 build and no GL needed
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
