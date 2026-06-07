#!/usr/bin/env bash
# Build the pxa_llama P100-only (sm_60) speed binary.
#
# This is the "build-speed" target: GGML_CUDA_F16 + FORCE_DMMV + MMV_Y2 leverage the P100's 2:1 FP16
# for k-quant dequant+matmul (~+19% over a stock build). All IQK kernels are preserved.
#
# DO NOT run this binary on a 1080 Ti — GGML_CUDA_F16=ON is catastrophic on its 1:64 FP16. For a mixed
# P100 + 1080 Ti split use build-multicard.sh instead.
#
# Prereq: apply the concurrent-hybrid fix first:
#   git clone https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp && git checkout 1520eda
#   git apply --3way /path/to/pxa_llama/patches/pxa_llama_v4_batched_delta_net.diff
#
# We build inside the CUDA 12.8 devel container (the resulting binary needs glibc 2.38 / GLIBCXX 3.4.31;
# build it where you'll run it, or in a matching runtime).
set -euo pipefail

SRC="${1:-$PWD}"                       # path to the ik_llama.cpp checkout (with the patch applied)
BUILD_DIR="${BUILD_DIR:-build-speed}"
JOBS="${JOBS:-$(nproc)}"
IMAGE="${IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"

echo ">> Configuring $SRC/$BUILD_DIR (sm_60, F16+FORCE_DMMV+MMV_Y2)"
docker run --rm --runtime=nvidia -v "$SRC":/work/src -w /work/src "$IMAGE" bash -lc "
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq cmake build-essential libgomp1 git >/dev/null 2>&1
  cmake -B $BUILD_DIR \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=60 \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FORCE_DMMV=ON \
    -DGGML_CUDA_MMV_Y=2 \
    -DGGML_SCHED_MAX_COPIES=4 \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build $BUILD_DIR --target llama-server -j$JOBS
"
echo ">> Built: $SRC/$BUILD_DIR/bin/llama-server"
echo ">> Run with: LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd (ik libs are scattered)"
echo ">> See launchers/ for ready-to-run invocations."
