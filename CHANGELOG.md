# Changelog — pxa_llama vs ik_llama.cpp

Exactly what we changed, and why. Base: `ikawrakow/ik_llama.cpp` (branch `pxa_llama`, cut against
commit `1520eda`).

## [v4] Concurrent‑hybrid Gated‑DeltaNet fix — the headline

**Problem.** Stock ik_llama corrupts hybrid / recurrent‑state models (qwen3next Gated‑DeltaNet:
Coder‑Next‑80B, Qwen3‑Next‑80B; qwen35moe: 35B/122B) at `np>=3`: concurrent slots bleed recurrent state →
garbage / cross‑conversation output. `np=2` is clean; `np>=3` is dirty; raising `np` makes it worse.

**Root cause (proven, see `docs/HYBRID-CONCURRENCY-BUG.md`).** `build_layer_attn_linear`'s mixed‑seq path
builds **N independent per‑token subgraphs**, each doing `get_rows`/`set_rows` on the persistent recurrent
pool `s_l[il]`. The ggml graph allocator (`ggml-alloc.c`) frees and **reuses buffer offsets** by
topological refcount; with N≥3 interleaved subgraphs a still‑live recurrent scratch gets reused →
cross‑sequence bleed. The per‑token‑loop *structure* is the bug — earlier fixes that only changed the
read/write *mechanism* (graph‑reuse guard, runtime gather, deferred scatter) each changed the failure mode
but never eliminated it.

**Fix (`PXA_LLAMA_FIX_v4`).** Replace the per‑token loop with **one batched multi‑seq delta‑net call**,
mirroring ik's own concurrency‑clean Mamba path (`graphs/build_mamba.cpp`): a single `ggml_ssm_conv` + a
single delta‑net over all tokens, with a `[n_kv, n_tokens]` sequence map routing inside the kernel and one
contiguous write‑back. ik's delta‑net CUDA kernel and the `ssm_conv` multi‑seq‑unique path already support
`n_seqs>1`; only the graph *builder* looped.

Files touched (see `patches/pxa_llama_v4_batched_delta_net.diff`, 387‑line diff):
- `src/llama-delta-net.cpp` — `build_qkv` now takes `state_row_idx`, `conv_seq_map`, `state_mask` and a
  `n_seqs_in` count instead of a build‑time scalar `state_seq_id_local`; batched gather/scatter of the
  recurrent state; `build_beta_gate` reshapes on `(n_seq_tokens, n_seqs)`; the per‑token caller loop is
  replaced by one batched core call.
- `src/llama.cpp` — fills the per‑seq state‑row map / conv seq map per step (the runtime routing tensors)
  instead of hardcoding seq 0.

**Result.** Distinct‑codeword cross‑bleed test: **CLEAN at np=4 and np=6** (stock = corrupt). Concurrent
tool‑call gauntlet 7/7 correct. np=1 speed unchanged. See `BENCHMARKS.md` §1.

### Evolution (banked in `patches/` history, for the curious)
- **v0** — blanket "never reuse graph for recurrent": proved the diagnosis (clean) but 0.69 tok/s (rebuilds
  every token). Unusable.
- **v1** — seq‑signature reuse key: restored np=1 speed, still dirty at np>1 (deeper than graph reuse).
- **v2** — runtime per‑seq `get_rows`/`set_rows` gather: np=1 clean, np>1 still dirty (per‑token in‑place
  aliasing).
- **v3 / v3b / v3c** — batched/deferred‑scatter attempts: v3b clean to np=2, dirty at np≥3 (the allocator
  hazard, finally pinned).
- **v4** — the single batched multi‑seq op. Clean to np≥6. **Shipped.**

## [build] Pascal sm_60 speed flags

Build the P100‑only binary with FP16‑path tuning that upstream stock images (sm_61+) don't enable:
`-DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_F16=ON -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_MMV_Y=2
-DGGML_SCHED_MAX_COPIES=4`, all IQK features preserved. ~+19% decode over a stock build, compiled in.
See `build/build-sm60.sh` and `docs/OLD-CARD-GUIDE.md`.

> Note: `GGML_CUDA_F16=ON` is a **P100‑only** win. It is catastrophic on the 1080 Ti (1:64 FP16). For a
> mixed P100 + 1080 Ti layer‑split, use a stock‑flags mainline/ik `sm_60;61` build instead — see
> `build/build-multicard.sh`.

## [docs / harnesses] Added

- `benchmarks/concurrency-test.sh` — the distinct‑codeword cross‑bleed acceptance test (the moat).
- `benchmarks/speed-test.sh` — single + aggregate‑concurrent tok/s probe.
- `docs/*` — root‑cause writeup, old‑card guide, compute‑bound measurement, MTP notes.
- `launchers/*` — ready‑to‑run server invocations.
- `decomp-router/*` — the companion decomposition router.

## Not changed

The IQK matmul/flash‑attention kernels, MoE/quant code, hybrid graph builders for the attention/MoE side,
and MTP plumbing are upstream ik_llama.cpp and are used as‑is. Non‑hybrid `qwen3moe` (e.g. 30B‑A3B) was
already concurrency‑clean on stock ik — we didn't touch it.
