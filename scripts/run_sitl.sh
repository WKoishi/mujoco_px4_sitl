#!/usr/bin/env bash
#
# Bring up PX4 SITL and the MuJoCo simulator together.
#
# Start order does not matter: we bind the HIL port before PX4 needs it, and PX4
# retries connect() every 500 us until we accept. We start first anyway, because
# PX4's boot blocks on our first HIL_SENSOR under lockstep.
#
# Ctrl-C stops both. PX4's stdout stays on this terminal so its shell (`commander
# status`, `listener sensor_baro`, `ekf2 status`) remains usable.
#
# --px4-only starts PX4 alone, for a caller that runs the simulator in its own
# process with an in-process controller (mujoco_px4_sitl.main.run).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${REPO_ROOT}/.." && pwd)"
PX4_DIR="${PX4_DIR:-${WORKSPACE}/PX4-Autopilot}"
VENV="${VENV:-${WORKSPACE}/.venv}"

AIRFRAME_ID="${PX4_SYS_AUTOSTART:-22001}"
INSTANCE="${PX4_INSTANCE:-0}"
MODEL="${MUJOCO_SITL_MODEL:-${REPO_ROOT}/models/quad_x.xml}"
SIM_ARGS=()
HEADLESS=1
PX4_ONLY=0

usage() {
	cat <<EOF
Usage: $(basename "$0") [options] [-- extra simulator args]

  -m, --model FILE     MuJoCo model (default: ${MODEL})
  -i, --instance N     PX4 instance; HIL port 4560+N (default: ${INSTANCE})
  -s, --speed FACTOR   simulated seconds per wall second (default: 1.0)
  -g, --gui            open the MuJoCo viewer
      --px4-only       start PX4 only; the simulator runs in the caller's process
  -h, --help           this message

Environment: PX4_DIR, VENV, PX4_SYS_AUTOSTART, MUJOCO_SITL_MODEL,
             PX4_HOME_LAT / PX4_HOME_LON / PX4_HOME_ALT.
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
		-m|--model) MODEL="$2"; shift ;;
		-i|--instance) INSTANCE="$2"; shift ;;
		-s|--speed) SIM_ARGS+=("--speed-factor" "$2"); shift ;;
		-g|--gui) HEADLESS=0 ;;
		--px4-only) PX4_ONLY=1 ;;
		-h|--help) usage; exit 0 ;;
		--) shift; SIM_ARGS+=("$@"); break ;;
		*) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
	esac
	shift
done

[ "${HEADLESS}" -eq 0 ] && SIM_ARGS+=("--viewer")
if [ "${PX4_ONLY}" -eq 1 ] && [ "${#SIM_ARGS[@]}" -gt 0 ]; then
	echo "error: --px4-only starts no simulator; give -s, -g and simulator" \
		"arguments to the process that runs it" >&2
	exit 2
fi

PX4_BIN="${PX4_DIR}/build/px4_sitl_default/bin/px4"
if [ ! -x "${PX4_BIN}" ]; then
	echo "error: ${PX4_BIN} not found." >&2
	echo "       source ${VENV}/bin/activate && make -C ${PX4_DIR} px4_sitl_default" >&2
	exit 1
fi

PYTHON="${VENV}/bin/python"
[ -x "${PYTHON}" ] || PYTHON="$(command -v python3)"

SIM_PID=""
PX4_PID=""

# TERM, then KILL after a grace period. Needed for PX4: its shutdown runs on the
# work queue, which under lockstep runs on simulated time, so once the simulator
# stops sending HIL_SENSOR, PX4 cannot finish exiting -- nor can a PX4 still
# blocked in boot. Waiting on it unbounded hangs here and orphans it.
stop() {
	local pid="$1" i
	{ [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; } || return 0
	kill -TERM "${pid}" 2>/dev/null || true
	for i in $(seq 30); do
		kill -0 "${pid}" 2>/dev/null || break
		sleep 0.1
	done
	if kill -0 "${pid}" 2>/dev/null; then
		echo "== PX4/simulator ${pid} did not exit in 3 s; killing it" >&2
		kill -KILL "${pid}" 2>/dev/null || true
	fi
	wait "${pid}" 2>/dev/null || true
}

cleanup() {
	trap - INT TERM EXIT
	# PX4 first, while the simulator still advances its clock.
	stop "${PX4_PID}"
	stop "${SIM_PID}"
}
trap cleanup INT TERM EXIT

if [ "${PX4_ONLY}" -eq 0 ]; then
	echo "== simulator: instance ${INSTANCE}, HIL port $((4560 + INSTANCE)), model ${MODEL}"
	PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		"${PYTHON}" -m mujoco_px4_sitl \
		--instance "${INSTANCE}" --model "${MODEL}" "${SIM_ARGS[@]}" &
	SIM_PID=$!

	# Let the socket bind before PX4's first connect attempt. Not required -- PX4
	# retries -- but it keeps the log clean.
	sleep 0.5
	if ! kill -0 "${SIM_PID}" 2>/dev/null; then
		echo "error: simulator exited during startup" >&2
		exit 1
	fi
else
	echo "== PX4 only: the simulator must bind HIL port $((4560 + INSTANCE)) itself"
fi

# PX4 wants to run from its build/rootfs directory.
ROOTFS="${PX4_DIR}/build/px4_sitl_default/rootfs"
mkdir -p "${ROOTFS}"

# Without a TTY the pxh shell redraws its prompt into the log endlessly, so use
# PX4's daemon mode there. Reach a daemonised instance with, from ${ROOTFS}:
#   ../bin/px4-commander status ; ../bin/px4-listener sensor_baro
# --px4-only always uses it: the terminal belongs to the caller's process.
PX4_FLAGS=(-i "${INSTANCE}")
if [ ! -t 1 ] || [ "${PX4_ONLY}" -eq 1 ]; then
	PX4_FLAGS+=(-d)
	echo "== PX4: daemon mode (-d); no pxh shell on this terminal"
fi

echo "== PX4: SYS_AUTOSTART=${AIRFRAME_ID} (boot blocks until our first HIL_SENSOR)"
(
	cd "${ROOTFS}"
	PX4_SYS_AUTOSTART="${AIRFRAME_ID}" \
		exec "${PX4_BIN}" "${PX4_FLAGS[@]}" \
		"${PX4_DIR}/build/px4_sitl_default/etc"
) &
PX4_PID=$!

# Exit as soon as either side goes down, so a crash is never silent.
if [ "${PX4_ONLY}" -eq 1 ]; then
	wait "${PX4_PID}" || true
	echo "== PX4 exited"
	exit 0
fi
while kill -0 "${SIM_PID}" 2>/dev/null && kill -0 "${PX4_PID}" 2>/dev/null; do
	sleep 0.5
done

kill -0 "${SIM_PID}" 2>/dev/null || echo "== simulator exited"
kill -0 "${PX4_PID}" 2>/dev/null || echo "== PX4 exited"
