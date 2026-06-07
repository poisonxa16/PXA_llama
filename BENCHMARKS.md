# pxa_llama — Benchmarks

Hardware: 1× **Tesla P100-PCIE-16GB** (Pascal sm_60: fast FP16 2:1, HBM2 ~732 GB/s, **NO int8/DP4A**),
1× **GTX 1080 Ti** (sm_61, 11GB, HAS DP4A). Unraid box. Engine: pxa_llama (ik_llama.cpp fork, build-speed,
sm_60 F16/FORCE_DMMV) for single-card; mainline llama.cpp (sm_60;61) for the 2-card layer-split.

Tag legend: **[M]** = directly measured · **[R]** = recorded from tuning sessions. Harnesses are in
[`benchmarks/`](benchmarks/). Decode on Pascal is **dequant/compute-bound, not bandwidth-bound** — see
[`docs/COMPUTE-BOUND-PASCAL.md`](docs/COMPUTE-BOUND-PASCAL.md); this is why so many "bandwidth" levers below
test NEUTRAL, and we report that honestly.

---

## 1. Correctness — concurrent hybrid (np>1) Gated-DeltaNet  ← the moat
Stock ik_llama corrupts hybrid/recurrent-state models (qwen3next Gated-DeltaNet / qwen35moe) at np>=3:
concurrent slots bleed recurrent state → garbage / cross-conversation output. pxa_llama's **v4
batched-delta-net fix** makes np>1 CLEAN.

| Engine | np=1 | np=2 | np=4 | np=6 |
|---|---|---|---|---|
| stock ik_llama (hybrid) | clean | clean | **CORRUPT (cross-bleed)** | CORRUPT |
| **pxa_llama (v4 fix)** | clean | clean | **CLEAN** | **CLEAN** |

[R] · Reproduce:
```bash
benchmarks/concurrency-test.sh http://127.0.0.1:8088 4 chat   # -> CLEAN on pxa_llama
benchmarks/concurrency-test.sh http://127.0.0.1:8088 6 chat   # -> CLEAN on pxa_llama; DIRTY on stock ik
```
- Test: distinct-codeword cross-bleed (each slot must return only its own codeword). pxa_llama: N=4
  (ZULU111/MANGO222/VELVET333/QUARTZ444) and N=6 (AAA01..FFF06) all unique, zero cross-bleed. [R]
- Concurrent tool-call gauntlet: 7/7 correct (distinct args, no cross-talk, no malformed). [R]

## 2. Single-card P100 speed (build + quant)
- pxa_llama F16/sm60/FORCE_DMMV/SCHED_MAX_COPIES build = **~+19% over stock** (build-time, compiled in). [R]
- `-ser 8,0.0` (10→8 experts) = +6%, np-clean, no tool-call quality loss. [R]
- **30B-A3B Q3_K_M (qwen3moe, non-hybrid), full-GPU:** ~**55-56 tok/s** single-stream, ~83 tok/s aggregate
  @ np4. Memory-bandwidth/dequant bound — KV-type/context/`-ser`/spec-decode/IQK-same-size all NEUTRAL. [M]
- **122B-A10B (qwen35moe) Q3, offloaded** `--n-cpu-moe 48`, single P100: **~18 tok/s** decode. [R]
- **80B Coder-Next Q3, offloaded**, single P100: **~25.7 tok/s** clean box (PCIe-wall; threads/mmap/mlock/
  NUMA/`-rtr`/lower-`--n-cpu-moe` = no gain, bandwidth-bound). [R]
- NEUTRAL on offloaded single card (tested): NUMA-pin, `-rtr`, MMV_Y2, IQK same-size quant, ngram-spec
  (also broke np-clean). [R]

## 3. MTP self-speculation (qwen35moe, np=1)  [M]
`--spec-type mtp:n_max=2,p_min=0.5` on a Claude-distilled Qwen3.6-35B-A3B-MTP Q2_K, P100, code prompt:

| config | tok/s | vs baseline | draft acceptance |
|---|---|---|---|
| baseline (no spec) | 43.3 | — | — |
| **MTP n_max=2** | **52.3** | **+21%** | 83% |
| MTP n_max=3 | 51.9 | +20% | 81% |

Lossless. SM% went *down* (93→89) — batched verify amortizes the dequant bottleneck. Only engages on GGUFs
that retain the `nextn` tensors; np=1 only (hurts concurrency). See [`docs/MTP.md`](docs/MTP.md).

## 4. Multi-GPU layer-split (P100 + 1080Ti) — opus-minime 35B-A3B Q2_K, full-GPU, np=8  [M]
Model split across both cards via mainline (sm_60;61), `--split-mode layer -ngl 99`, no CPU offload.
- **Single-stream baseline: ~46 tok/s** (1080Ti-favored split) vs ~13–18 offloaded — full-GPU is the fast regime.
- **DP4A finding:** Q2_K is 2-bit; the 1080Ti (sm_61, DP4A) computes low-bit FASTER per-layer than the P100
  (sm_60, no DP4A) despite less bandwidth → **more layers on the 1080Ti is faster**.
  - 1080Ti-favored (`-ts 8,16`): **46 tok/s, 2.18x batching**
  - P100-favored (`-ts 2,1`): 40 tok/s, 1.95x (slower)
- **Batching curve (concurrent wall-clock speedup):**

  | N (concurrent) | speedup |
  |---|---|
  | 2 | ~1.8x |
  | **4** | **2.18x** (peak) |
  | 6 | 2.11x |
  | 8 | 1.91x |
  → the 2-card rig **saturates at N≈4**; beyond that, streams contend and batching drops.

A separate fair-fight (35B-heretic-MTP Q4_K_M, both cards, mainline, draft-mtp n3, f16 KV, P100-heavy split)
hit **~60-63 tok/s** — and that's a FLOOR (a gimped-FP16 1080Ti drags it; identical P100s would be cleaner).

## 5. opus-microme — the in-engine decomposition model  [M]
Purpose-built dependency cross-encoder (given prompt + clauseA + clauseB → "does B depend on A?").
Held-out gauntlets (hand-written, never trained on): atomic / parallel / dependent / mixed / adversarial
"looks-independent-but-isn't" + a Spanish/English-trap set.

| Model | testset (32 EN) | testset2 (43 hard EN) | testset3 (25 ES+EN) | dangerous FP | size (int8) | latency (CPU) |
|---|---|---|---|---|---|---|
| ELECTRA-small (EN only) | 100% | 98% | — | **0** | 13.8 MB | **4.4 ms/pair** |
| **distilbert-en-es (bilingual, shipped)** | **100%** | **98%** | **100%** | **0** | **68 MB** | **~10.9 ms/pair** |
| granite4:micro (R&D stand-in) | — | — | — | — | (3B) | ~1400 ms |
| cortex 35B self-decompose | — | — | — | — | (35B) | ~6600 ms |
- **Aggregate: 99/100 held-out edge cases, ZERO dangerous false-parallels** (incl. comma/quote/binomial
  traps AND Spanish hidden dependencies). 1 conservative miss (a fan-in case).
- Pareto note: pushing the last miss either reintroduced a dangerous chain false-parallel or cost ≥1 other
  conservative miss → 99/100 @ 0-dangerous is the shipped operating point.

## 6. C++ in-engine decomposer (`llama-decompose-server`)  [M]
The 68MB model converted to GGUF (BERT body) + a 2-layer head, embedded in a llama.cpp engine binary
(seg + ggml embed + head matmul + DAG, no ONNX/Python at runtime). Built CPU-only, isolated from prod.
- **Validated vs GOLD: 99/100, 0 dangerous** — bit-identical accuracy to the ONNX model, pure ggml.
- Latency: **9 ms** atomic, **65–80 ms** multi-clause (per-pair on CPU; batchable).
- **Size gate** shipped: emits `recommend` (≥2 independent *substantial* sub-tasks) separate from the
  structural `fanout` → only fan out when it pays.

## 7. Fan-out vs linear — end-to-end (multi-GPU, the Skeleton-of-Thought validation)  [M]
Decompose via the in-engine decomposer → fan out independent sub-tasks across cortex slots → vs one linear call.
| Workload | Linear | Fan-out | Speedup |
|---|---|---|---|
| Short conversational (joke/movie/book) | fast (model answers concisely) | 3× prefill + token-bloat | **0.2–0.5x ✗** |
| **Substantial, 3 tasks** (~150-word explanations) | 11.9 s @ 48 t/s | 6.9 s @ 81 t/s agg | **1.73x ✓** |
| Substantial, 4 tasks | 11.8 s | 9.0 s | 1.30x |
| Substantial, 5–6 tasks | — | — | noisy / regresses (token-imbalance) |
- **Verdict:** fan-out wins **1.73x on substantial 3-way parallel work** (≈80% of the 2.18x batching ceiling)
  and correctly loses (now gated out) on short prompts. Win is real only on **under-utilized rigs** (idle
  slots) — a busy multi-tenant server is already batch-saturated, so this is a homelab/personal-agent win.

## Methodology / honesty
- Prior art: the fan-out is a tuned, gated, auto + validated **Skeleton-of-Thought** (Ning et al.,
  arXiv [2307.15337](https://arxiv.org/abs/2307.15337), up to 2.39x). Our adds: a sub-10ms purpose-built
  in-engine decomposer (vs using the big LLM for the skeleton), a dependency-DAG safety layer (never
  false-parallelize), a size gate, bilingual 99/100 validation.
- All [M] numbers were measured on the box; [R] are recorded prior-session findings. Single-card decode is
  dequant/compute-bound (SM ~95-99% pegged, HBM2 bus ~16% idle during decode); batching/fan-out/the 2:1
  FP16 help the compute-bound regimes (concurrency, prefill, dequant, rerankers). F16 *storage* of large
  models is slower on Pascal (more bytes); the FP16 win is the *compute* path + concurrency.
