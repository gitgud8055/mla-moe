#!/usr/bin/env bash
#
# install_rocprof_compute.sh
# --------------------------
# Install the ROCm Compute Profiler (rocprof-compute) FROM SOURCE into a ROCm
# container. Run it directly inside the container, with no editing required:
#
#     bash docs/install_rocprof_compute.sh
#
# When it finishes, you can immediately run:
#
#     rocprof-compute --version
#
# Reference:
#   https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/source-install.html
#
# This script is idempotent (safe to re-run). Set FORCE=1 to reinstall from
# scratch. Change the install location with INSTALL_DIR (default:
# /opt/rocprofiler-compute).
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (sensible defaults, no need to change)
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/opt/rocprofiler-compute}"
SRC_DIR="${SRC_DIR:-$INSTALL_DIR/src}"
REPO="${ROCPROF_COMPUTE_REPO:-https://github.com/ROCm/rocm-systems.git}"
BRANCH="${ROCPROF_COMPUTE_BRANCH:-develop}"
SUBPROJ="projects/rocprofiler-compute"

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'    "$*"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n'   "$*" >&2; }

# ---------------------------------------------------------------------------
# 0. Root privileges (needed to install into /opt and apt-get libdw-dev)
# ---------------------------------------------------------------------------
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    err "Must run as root (or have sudo). Inside a container you are usually root already."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 1. Check required build tools
# ---------------------------------------------------------------------------
log "Checking build tools..."
missing=""
for t in git cmake make gcc python3; do
  command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
done
if [ -n "$missing" ]; then
  err "Missing tools:$missing . Please install them and re-run."
  exit 1
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  err "pip for python3 is missing (python3 -m pip)."
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. System dependency: libdw (not mentioned in the guide, but required by
#    CMake)
# ---------------------------------------------------------------------------
if pkg-config --exists libdw 2>/dev/null; then
  log "libdw is already present."
else
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing libdw-dev (elfutils) via apt..."
    $SUDO apt-get update -y
    $SUDO apt-get install -y libdw-dev
  else
    err "apt-get not found and libdw is missing. Please install the elfutils/libdw dev package manually."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 3. Skip if already installed (unless FORCE=1)
# ---------------------------------------------------------------------------
if [ "${FORCE:-0}" != "1" ] && [ -x /usr/local/bin/rocprof-compute ] \
   && /usr/local/bin/rocprof-compute --version >/dev/null 2>&1; then
  log "rocprof-compute is already installed. (Set FORCE=1 to reinstall.)"
  /usr/local/bin/rocprof-compute --version || true
  exit 0
fi

# ---------------------------------------------------------------------------
# 4. Fetch source (sparse checkout, only rocprofiler-compute to keep it light)
# ---------------------------------------------------------------------------
$SUDO mkdir -p "$INSTALL_DIR"
if [ -d "$SRC_DIR/.git" ]; then
  log "Updating existing source at $SRC_DIR..."
  $SUDO git -C "$SRC_DIR" fetch --depth=1 origin "$BRANCH"
  $SUDO git -C "$SRC_DIR" checkout "$BRANCH"
  $SUDO git -C "$SRC_DIR" reset --hard "origin/$BRANCH"
else
  log "Cloning $REPO (sparse: $SUBPROJ, branch: $BRANCH)..."
  $SUDO rm -rf "$SRC_DIR"
  $SUDO git clone --no-checkout --filter=blob:none "$REPO" "$SRC_DIR"
  $SUDO git -C "$SRC_DIR" sparse-checkout init --cone
  $SUDO git -C "$SRC_DIR" sparse-checkout set "$SUBPROJ"
  $SUDO git -C "$SRC_DIR" checkout "$BRANCH"
fi

PROJ_DIR="$SRC_DIR/$SUBPROJ"
VERSION="$($SUDO cat "$PROJ_DIR/VERSION" | tr -d '[:space:]')"
log "rocprof-compute version: $VERSION"

# ---------------------------------------------------------------------------
# 5. Install Python deps into a separate directory (python-libs). This avoids
#    numpy conflicts with vLLM/torch in the container: requirements currently
#    pin numpy 1.26.4 while containers commonly ship numpy 2.x.
# ---------------------------------------------------------------------------
log "Installing Python dependencies into $INSTALL_DIR/python-libs (isolated)..."
$SUDO python3 -m pip install --no-cache-dir \
  -t "$INSTALL_DIR/python-libs" \
  -r "$PROJ_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 6. Configure, build, and install
# ---------------------------------------------------------------------------
log "Configuring CMake..."
$SUDO rm -rf "$PROJ_DIR/build"
$SUDO mkdir -p "$PROJ_DIR/build"
$SUDO cmake -S "$PROJ_DIR" -B "$PROJ_DIR/build" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/$VERSION" \
  -DPYTHON_DEPS="$INSTALL_DIR/python-libs" \
  -DMOD_INSTALL_PATH="$INSTALL_DIR/modulefiles/rocprofiler-compute"

log "Building + installing..."
$SUDO make -C "$PROJ_DIR/build" -j"$(nproc)" install

# ---------------------------------------------------------------------------
# 7. Create a wrapper in /usr/local/bin so rocprof-compute works immediately.
#    The wrapper sets PYTHONPATH to the isolated python-libs directory.
# ---------------------------------------------------------------------------
log "Creating wrapper /usr/local/bin/rocprof-compute..."
$SUDO tee /usr/local/bin/rocprof-compute >/dev/null <<EOF
#!/usr/bin/env bash
# Auto-generated by install_rocprof_compute.sh
export PYTHONPATH="$INSTALL_DIR/python-libs\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$INSTALL_DIR/$VERSION/bin/rocprof-compute" "\$@"
EOF
$SUDO chmod +x /usr/local/bin/rocprof-compute

# Backup for interactive shells (for modulefiles or a manual PATH).
$SUDO tee /etc/profile.d/rocprof-compute.sh >/dev/null <<EOF
# Auto-generated by install_rocprof_compute.sh
export PATH="$INSTALL_DIR/$VERSION/bin:\$PATH"
export PYTHONPATH="$INSTALL_DIR/python-libs\${PYTHONPATH:+:\$PYTHONPATH}"
EOF

# ---------------------------------------------------------------------------
# 8. Verify
# ---------------------------------------------------------------------------
log "Verifying installation:"
rocprof-compute --version

cat <<'EOF'

============================================================================
  INSTALLATION SUCCESSFUL  ->  run 'rocprof-compute' from anywhere.
----------------------------------------------------------------------------
  Basic profile + CLI analysis:

    rocprof-compute profile -n my_app \
      --output-directory ./workloads --overwrite -- ./my_app <args>
    rocprof-compute analyze -p ./workloads > analyze.out 2>&1

  Example GEMM profile:

    rocprof-compute profile -n gemm_fma -- ./gemm_fma 1024

  Profile while skipping an application's reference path:

    SKIP_REF=1 rocprof-compute profile \
      --output-directory ./profile-rocprofv3/lds \
      -n gemm_lds -- ./gemm_mfma02_lds 6016

  Analysis modes:

    # CLI (default)
    rocprof-compute analyze -p ./profile-rocprofv3/lds

    # TUI
    rocprof-compute analyze -p ./profile-rocprofv3/lds \
      --experimental --tui

    # GUI
    rocprof-compute analyze -p ./profile-rocprofv3/lds \
      --experimental --gui

    # Compare two workloads
    rocprof-compute analyze -p <workload_1_dir> -p <workload_2_dir>

  Use --output-directory for profile output; do not use --path.
============================================================================
EOF
