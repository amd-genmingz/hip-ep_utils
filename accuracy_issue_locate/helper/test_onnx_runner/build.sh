#!/usr/bin/env bash
# Build test_onnx_runner (RelWithDebInfo) into ./build.

# ---- user settings ---------------------------------------------------------
# ONNX Runtime install root, containing include/ and lib/.
ONNXRUNTIME_ROOT=""
# CMake generator, e.g. "Ninja". Empty uses the CMake default.
GENERATOR=""
# ----------------------------------------------------------------------------

set -euo pipefail

if [ -z "$ONNXRUNTIME_ROOT" ]; then
  echo "ERROR: set ONNXRUNTIME_ROOT at the top of $0" >&2
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SRC_DIR/build"

GEN_ARGS=()
if [ -n "$GENERATOR" ]; then
  GEN_ARGS=(-G "$GENERATOR")
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build "$BUILD_DIR" --config RelWithDebInfo --parallel
