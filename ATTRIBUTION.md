# Attribution & Provenance

pxa_llama is **not** a from‑scratch inference engine. It is a fork with a focused set of patches.

## Lineage

```
ggml-org/llama.cpp   (MIT, the original)
        │
        ▼
ikawrakow/ik_llama.cpp   (MIT, fork — custom IQK CUDA matmul/flash-attention kernels, MoE/quant work,
        │                 hand-rolled hybrid graph builders for qwen3next / qwen35moe, MTP plumbing)
        ▼
pxa_llama   (this project — our patches ON TOP of ik_llama.cpp)
```

Essentially everything that makes the engine *fast* — the IQK matmul kernels, the flash‑attention,
the MoE handling, the quant formats, the hybrid graph builders, the MTP plumbing — comes from
**ik_llama.cpp** and **llama.cpp**. We stand entirely on that work.

## What is *ours* (the delta)

1. **The concurrent‑hybrid (np>1) Gated‑DeltaNet fix** — a batched, multi‑seq delta‑net path replacing
   ik's per‑token loop, so concurrent slots no longer bleed recurrent state. This is the headline
   contribution. (`patches/pxa_llama_v4_batched_delta_net.diff`)
2. **Pascal sm_60 build tuning** — `GGML_CUDA_F16=ON` + `GGML_CUDA_FORCE_DMMV=ON` (+`MMV_Y=2`) +
   `SCHED_MAX_COPIES`, targeting the P100 that upstream images don't build for.
3. **Documentation & measurement** — the old‑card FP16/DP4A economics guide, the compute‑bound
   measurement, the per‑model max‑settings, and reproducible benchmarks.
4. **The decomposition router** (companion) — a purpose‑built bilingual dependency cross‑encoder +
   gated fan‑out, an implementation of Skeleton‑of‑Thought tailored for idle old‑card rigs.

See `CHANGELOG.md` for the exact, file‑level deltas.

## Licensing

All three layers are MIT. The combined work remains MIT. See `LICENSE`. We keep the upstream
copyright notices intact and add our own; we make no claim over upstream code.

## How to get the full source

pxa_llama is distributed as **patches on top of ik_llama.cpp** (so the delta is auditable and the
upstream attribution unambiguous). To build the full engine:

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp && git checkout 1520eda
git apply --3way /path/to/pxa_llama/patches/pxa_llama_v4_batched_delta_net.diff
```

Then build with `build/build-sm60.sh`. See `patches/README.md` and `build/README.md`.

## Upstream projects — please star them

- ik_llama.cpp — https://github.com/ikawrakow/ik_llama.cpp
- llama.cpp — https://github.com/ggml-org/llama.cpp
