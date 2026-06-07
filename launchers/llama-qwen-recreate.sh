#!/bin/bash
# PRODUCTION den: Qwen3.5-122B-A10B abliterated + MTP (qwen35moe hybrid) on pxa_llama build-speed.
# Smart-tier den (2026-06-06 tuning verdict). Single P100 OFFLOADED (~18 tok/s); shines GPU-resident on the rig.
# Config: reasoning-capable (thinking on by default via --jinja; orchestrator sends big max_tokens +
# handles reasoning_content), MTP n3 self-spec, q8 KV, np=2 (=8192/slot, clears the orchestrator 4096-slot 400 bug).
set -u
NAME=llama-qwen; GPU=GPU-aad5ef40-9b80-8fd0-4391-dfe595f42640
MODEL=/mtp/122B-Aggressive-Q3_K_M-MTP.gguf
BUILD=/mnt/user/appdata/ik_llama/ik_llama.cpp/build-speed
if docker ps --filter name=$NAME --format "{{.Names}}" | grep -q "^$NAME$"; then exit 0; fi
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --restart unless-stopped --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=$GPU \
  -e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd \
  -p 8088:8080 -v $BUILD:/build:ro -v /mnt/cachetwo/models/qwen/mtp:/mtp:ro \
  --health-cmd "bash -c 'echo > /dev/tcp/localhost/8080'" --health-interval 30s --health-timeout 10s \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 /build/bin/llama-server -m "$MODEL" \
  -c 16384 -ngl 99 --n-cpu-moe 48 -np 2 -fa on -ctk q8_0 -ctv q8_0 \
  --spec-type "mtp:n_max=3,p_min=0.5" \
  --jinja --temp 0.5 --top-p 0.8 --top-k 20 --min-p 0.0 --host 0.0.0.0 --port 8080
