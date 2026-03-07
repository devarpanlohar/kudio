#!/usr/bin/env bash
# build.sh — compile the kudio C++ DSP engine
# ═════════════════════════════════════════════
# Run from the kudio/ project root:
#
#   chmod +x build.sh && ./build.sh
#
# After a successful build, kudio_dsp.so (or .pyd on Windows) appears in
# this directory and is importable by the Python modules.
#
# Requirements: Python 3.10+, pip, cmake >= 3.14, a C++17 compiler.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════"
echo "  Kudio C++ DSP Engine — Build Script"
echo "═══════════════════════════════════════════"

# ── 1. Python dependencies ────────────────────────────────────────────────────
echo "[1/4] Installing Python build dependencies..."
pip install pybind11 numpy sounddevice scipy yt-dlp \
    --break-system-packages -q

# ── 2. Create build directory ─────────────────────────────────────────────────
echo "[2/4] Configuring CMake (Release build, march=native)..."
cd cpp
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF

# ── 3. Compile ────────────────────────────────────────────────────────────────
echo "[3/4] Compiling C++ extension (this takes ~10 seconds)..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
make -j"$NPROC"

# ── 4. Copy .so to project root ───────────────────────────────────────────────
echo "[4/4] Installing kudio_dsp.so → project root..."
SO=$(find . -name "kudio_dsp*.so" -o -name "kudio_dsp*.pyd" 2>/dev/null | head -1)
if [ -z "$SO" ]; then
    echo "ERROR: build produced no .so file — check compiler output above."
    exit 1
fi
cp "$SO" "$SCRIPT_DIR/"

echo ""
echo "✓  Build complete: $SCRIPT_DIR/$(basename "$SO")"
echo ""
echo "  Run:  python main.py"
echo "═══════════════════════════════════════════"