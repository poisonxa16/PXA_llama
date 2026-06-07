# Launchers

Ready‑to‑run `llama-server` invocations (the docker wrapper + the scattered‑lib `LD_LIBRARY_PATH` + the
verified per‑model flags). Adapt paths, GPU UUID, and model file to your box.

## `llama-qwen-recreate.sh`
Production‑style recreate script for a hybrid smart‑tier den (shown: 122B‑A10B qwen35moe + MTP, offloaded on
one P100). Has a `docker ps` guard so it's a no‑op if already running. Key points it demonstrates:
- Image `nvidia/cuda:12.8.1-devel-ubuntu24.04`, `--runtime=nvidia`, build dir mounted read‑only at `/build`.
- `LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd` (ik libs are scattered).
- `--spec-type mtp:n_max=3,p_min=0.5` (only engages if the GGUF retains `nextn` tensors — see
  [`../docs/MTP.md`](../docs/MTP.md)).
- `--n-cpu-moe 48` to offload experts to CPU (a 122B does not fit 16 GB).

## `opus-final-battery.sh`
The full tuning/validation battery used to lock the qwen35moe (35B‑A3B "opus‑minime") daily‑driver config:
sweeps configs, runs coding/tool‑call/reasoning probes, checks MTP acceptance and VRAM. Useful as a template
for validating your own model + flags before putting it in production.

## The per‑model flags
The verified "max settings" tables (30B / 80B / 122B / 35B) live in
[`../docs/OLD-CARD-GUIDE.md`](../docs/OLD-CARD-GUIDE.md). Start there, then drop the chosen flags into a
launcher.

> These scripts reference example box paths (`/mnt/...`) and a specific GPU UUID — they're illustrations,
> not turnkey for your machine. Edit `MODEL=`, the GPU id, and the mount paths.
