#!/usr/bin/env bash
# Build a mixed-arch (P100 sm_60 + GTX 1080 Ti sm_61) binary for a 2-card layer-split.
#
# IMPORTANT: do NOT enable GGML_CUDA_F16 here. F16=ON is the P100 speed win but is catastrophic on the
# 1080 Ti's 1:64 FP16. For a span of both cards, stock flags + DP4A on the 1080 Ti is the right tool. The
# 1080 Ti's DP4A actually makes it *faster* per-layer for low-bit (Q2_K) quants — favor it in the split
# (e.g. -ts 8,16). See docs/OLD-CARD-GUIDE.md.
#
# You can build either mainline llama.cpp or ik_llama.cpp here; mainline is concurrency-correct for hybrids
# out of the box (the pxa_llama fix targets the ik single-card build), and spans both arches cleanly.
set -euo pipefail

SRC="${1:-$PWD}"
BUILD_DIR="${BUILD_DIR:-build-multicard}"
JOBS="${JOBS:-$(nproc)}"
IMAGE="${IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"

echo ">> Configuring $SRC/$BUILD_DIR (sm_60;61, stock flags, NO F16)"
docker run --rm --runtime=nvidia -v "$SRC":/work/src -w /work/src "$IMAGE" bash -lc "
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq cmake build-essential libgomp1 git >/dev/null 2>&1
  cmake -B $BUILD_DIR \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES='60;61' \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build $BUILD_DIR --target llama-server -j$JOBS
"
echo ">> Built: $SRC/$BUILD_DIR/bin/llama-server"
echo ">> Layer-split example (Q2_K, favor the DP4A 1080 Ti):"
echo "   ./llama-server -m model-Q2_K.gguf --split-mode layer -ngl 99 -ts 8,16 --main-gpu 1 -np 8 -fa on"
