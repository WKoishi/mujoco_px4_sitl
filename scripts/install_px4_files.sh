#!/usr/bin/env bash
#
# Install (or remove) a PX4 airframe file and its .post into a PX4 tree.
#
# Defaults to this repository's quad; --airframe installs one from anywhere,
# which is how a generated airframe that lives with its private model gets in.
# Airframes coexist in the tree -- select one at boot with PX4_SYS_AUTOSTART.
#
# Copying the files is NOT sufficient. ROMFS contents are enumerated explicitly
# in init.d-posix/airframes/CMakeLists.txt -- one literal filename per line, with
# .post files listed separately, and no glob. An unregistered airframe is not
# packaged into the built ROMFS and PX4 fails at boot with
# "Error: no autostart file found".
#
# So this script copies the files AND registers both in that CMakeLists. Both
# steps are idempotent, and --uninstall reverses both. The PX4 checkout ends up
# carrying exactly one patched file; keep it that way.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$(cd "${REPO_ROOT}/../PX4-Autopilot" 2>/dev/null && pwd || true)}"
MARKER="# [22000, 22999] Reserve for custom models"

# Default: this repo's quad. --airframe takes a path instead, which is how a
# generated airframe living outside the repo gets installed -- the X8's CA_ROTOR*
# block is hub geometry, so it stays with the private model (AGENTS.md 4).
AIRFRAME_SRC="${REPO_ROOT}/px4/22001_mujoco_quad"

usage() {
	cat <<EOF
Usage: $(basename "$0") [--uninstall] [--px4-dir DIR] [--airframe PATH]

Installs an airframe file and its .post into a PX4 tree and registers both
in the ROMFS airframe CMakeLists.

  --uninstall      remove the files and their CMakeLists entries
  --px4-dir DIR    PX4-Autopilot checkout (default: \$PX4_DIR or ../PX4-Autopilot)
  --airframe PATH  airframe to install (default: px4/22001_mujoco_quad). May
                   live outside this repo: a generated X8 airframe sits with its
                   private model. The .post comes from this repo either way and
                   is copied under the airframe's own name, because rcS looks
                   for "\$autostart_file".post and the name must follow.

After installing, rebuild so the ROMFS is repackaged (seconds, not minutes):
  source .venv/bin/activate && make -C "\$PX4_DIR" px4_sitl_default
EOF
}

UNINSTALL=0
while [ $# -gt 0 ]; do
	case "$1" in
		--uninstall) UNINSTALL=1 ;;
		--px4-dir) PX4_DIR="$2"; shift ;;
		--airframe) AIRFRAME_SRC="$2"; shift ;;
		-h|--help) usage; exit 0 ;;
		*) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
	esac
	shift
done

AIRFRAME="$(basename "${AIRFRAME_SRC}")"
# PX4 derives SYS_AUTOSTART from the filename, and the CMakeLists marker this
# script inserts after reserves 22000-22999. A name outside that pattern would
# register into the wrong range or not boot at all.
case "${AIRFRAME}" in
	22[0-9][0-9][0-9]_*) ;;
	*)
		echo "error: airframe name must be 22NNN_<name>, since the id comes" >&2
		echo "       from the filename and 22000-22999 is the reserved range." >&2
		echo "       Got: ${AIRFRAME}" >&2
		exit 1 ;;
esac

# The .post is platform-independent -- it starts the three sensor_*_sim modules
# that px4-rc.mavlinksim leaves out -- so every airframe shares this repo's copy,
# installed under its own name.
POST_SRC="${REPO_ROOT}/px4/22001_mujoco_quad.post"

if [ -z "${PX4_DIR}" ] || [ ! -d "${PX4_DIR}" ]; then
	echo "error: PX4 tree not found. Pass --px4-dir or set PX4_DIR." >&2
	exit 1
fi

AIRFRAME_DIR="${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes"
CMAKELISTS="${AIRFRAME_DIR}/CMakeLists.txt"

if [ ! -f "${CMAKELISTS}" ]; then
	echo "error: not a PX4 tree (missing ${CMAKELISTS})" >&2
	exit 1
fi

register() {
	local name="$1"
	if grep -qx "[[:space:]]*${name}" "${CMAKELISTS}"; then
		echo "  already registered: ${name}"
		return
	fi
	if ! grep -qF "${MARKER}" "${CMAKELISTS}"; then
		echo "error: marker line not found in ${CMAKELISTS}." >&2
		echo "       PX4 may have reorganised it; register ${name} by hand." >&2
		exit 1
	fi
	# Insert after the reserved-range comment, tab-indented like its neighbours.
	awk -v marker="${MARKER}" -v entry="${name}" '
		{ print }
		index($0, marker) { print "\t" entry }
	' "${CMAKELISTS}" >"${CMAKELISTS}.tmp"
	mv "${CMAKELISTS}.tmp" "${CMAKELISTS}"
	echo "  registered: ${name}"
}

unregister() {
	local name="$1"
	if ! grep -qx "[[:space:]]*${name}" "${CMAKELISTS}"; then
		echo "  not registered: ${name}"
		return
	fi
	grep -vx "[[:space:]]*${name}" "${CMAKELISTS}" >"${CMAKELISTS}.tmp"
	mv "${CMAKELISTS}.tmp" "${CMAKELISTS}"
	echo "  unregistered: ${name}"
}

if [ "${UNINSTALL}" -eq 1 ]; then
	echo "Removing ${AIRFRAME} from ${PX4_DIR}"
	# Longest name first so the .post entry is not matched by the bare name.
	unregister "${AIRFRAME}.post"
	unregister "${AIRFRAME}"
	for f in "${AIRFRAME}" "${AIRFRAME}.post"; do
		if [ -e "${AIRFRAME_DIR}/${f}" ]; then
			rm -f "${AIRFRAME_DIR}/${f}"
			echo "  removed: ${f}"
		fi
	done
	echo "Done. Rebuild to repackage the ROMFS."
	exit 0
fi

if [ ! -f "${AIRFRAME_SRC}" ]; then
	echo "error: airframe not found: ${AIRFRAME_SRC}" >&2
	exit 1
fi
if [ ! -f "${POST_SRC}" ]; then
	echo "error: .post not found: ${POST_SRC}" >&2
	exit 1
fi

echo "Installing ${AIRFRAME} into ${PX4_DIR}"
install -m 0755 "${AIRFRAME_SRC}" "${AIRFRAME_DIR}/${AIRFRAME}"
echo "  copied: ${AIRFRAME}  (from ${AIRFRAME_SRC})"
install -m 0755 "${POST_SRC}" "${AIRFRAME_DIR}/${AIRFRAME}.post"
echo "  copied: ${AIRFRAME}.post  (from $(basename "${POST_SRC}"))"
register "${AIRFRAME}"
register "${AIRFRAME}.post"

cat <<EOF
Done. Next:
  source "${REPO_ROOT}/../.venv/bin/activate"   # PX4's build needs this venv
  make -C "${PX4_DIR}" px4_sitl_default
Then confirm both files landed in the built ROMFS:
  ls "${PX4_DIR}/build/px4_sitl_default/etc/init.d-posix/airframes/" | grep ${AIRFRAME%%_*}
Boot this airframe with:
  PX4_SYS_AUTOSTART=${AIRFRAME%%_*} scripts/run_sitl.sh ...
EOF
