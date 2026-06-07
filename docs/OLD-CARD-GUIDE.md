# Old‑card guide — Pascal P100 / GTX 1080 Ti FP16, DP4A & quant tuning

How to actually get good throughput out of cheap, old datacenter GPUs. The short version: **know which
compute path your card is fast at, and pick your build flags, quant, and split to match.**

## The two cards, and why they behave differently

| | Tesla P100 (GP100, sm_60) | GTX 1080 Ti (GP102, sm_61) |
|---|---|---|
| FP16 throughput | **2:1 (fast)** | 1:64 (catastrophic) |
| int8 / DP4A | **NO** | **YES** |
| Memory | 16 GB HBM2, ~732 GB/s | 11 GB GDDR5X, ~484 GB/s |
| Best at | FP16 compute, big models, KV in f16 | low‑bit quant (Q2_K/Q4) matmul via DP4A |

These differences drive every decision below.

`fast_fp16_available(cc) = cc>=600 && cc!=610` → **P100 (600) = TRUE, 1080 Ti (610) = FALSE.** The P100 is
the fast‑FP16 card; the 1080 Ti is the fast‑int8 card.

## Build flags

### P100‑only binary (the pxa_llama "build‑speed" target)
```
-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
-DGGML_CUDA_F16=ON -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_MMV_Y=2 \
-DGGML_SCHED_MAX_COPIES=4
```
- `GGML_CUDA_F16=ON` + `FORCE_DMMV` route k‑quant decode onto the half2 HFMA2 DMMV path, using the P100's
  2:1 FP16 instead of the sm_60 scalar int8 fallback. **~+19% over a stock build**, compiled in.
- All `GGML_IQK_*` features (matmul, flash‑attention) stay ON — that's ik's speed; don't disable them.
- **Do NOT put this binary on a 1080 Ti.** `F16=ON` is suicide on 1:64 FP16 hardware.

See [`../build/build-sm60.sh`](../build/build-sm60.sh).

### Mixed P100 + 1080 Ti (layer‑split)
Use a **stock‑flags** `sm_60;61` build (mainline llama.cpp or ik without F16), because `GGML_CUDA_F16=ON`
would wreck the 1080 Ti. The 1080 Ti's DP4A makes it good for the low‑bit layers anyway. See
[`../build/build-multicard.sh`](../build/build-multicard.sh).

## The compute‑bound truth (measured)

On the P100, **decode is dequant/compute‑bound, not bandwidth‑bound.** During sustained 30B‑A3B Q3_K_M
decode we measured **SM util 95–99% (pegged)** and **memory‑controller util ~15–21% (mostly idle)**. The
SMs spend ~96% of their time unpacking Q3_K superblocks (no DP4A → scalar ALU path); the HBM2 bus sits
idle because the cores can't consume data fast enough. Full writeup:
[`COMPUTE-BOUND-PASCAL.md`](COMPUTE-BOUND-PASCAL.md).

**Consequence:** the usual "bandwidth" levers are **NEUTRAL** on one P100 — we tested and confirmed:
KV‑type (q8/q4), context size, `-ser` expert reduction, `-ub`/batch, IQK same‑size quants, threads, mmap,
mlock, NUMA pin, `-rtr`, MMV_Y2 (on offloaded). What *does* help: the F16/FORCE_DMMV build, MTP
self‑spec (compute‑side amortization, np=1), and **more cards**.

## Quant choice on Pascal
- **Q3_K_M** is the sweet spot for 16 GB: fits big MoEs, decent quality. **Q4_K_M** is a near‑free quality
  bump when you have the byte headroom (decode speed ~flat — you're dequant‑bound, not byte‑bound).
- **IQK same‑size quants (IQ3_KS) gave ZERO speed gain** vs Q3_K_M on the P100 — same bytes/token, and the
  ik dequant‑kernel edge doesn't materialize here. Not worth re‑quantizing.
- **2‑bit (Q2_K / IQ2_*)** reads fewer bytes (faster) but tanks coding/tool‑call reliability — only use it
  on the 1080 Ti split where DP4A makes it fast *and* you accept the quality hit (or for the decomposer).

## Per‑model max settings (verified)

### 30B‑A3B (qwen3moe, non‑hybrid) — the fast daily driver, full‑GPU on one P100
```
-c 32768 -ngl 99 -np 4 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0
```
- **54–56 tok/s** single‑stream, ~83 tok/s aggregate @ np4, clean concurrency. VRAM ~16.1/16.4 GB.
- For more context at ~same speed: `-c 65536 -ctk q4_0 -ctv q4_0` (16384/slot, slight KV‑quality tradeoff).
- 128k does **not** fit no‑offload on 16 GB; ceiling ~64k via q4 KV.

### 80B Coder‑Next (qwen3next HYBRID) — smart escalation, offloaded on one P100
```
-c 65536 -ngl 99 -np 4 -fa on -ctk q8_0 -ctv q8_0 --n-cpu-moe 33 -ser 8,0.0 --jinja
```
- **~25.7 tok/s** single‑stream (clean box), np‑CLEAN to 7 (needs the v4 fix!), tool‑calls verified.
- `--n-cpu-moe` is **not** a speed lever here (33/31/29 identical; 27 OOMs) — the cost is the per‑token
  GPU↔CPU round‑trip, not the layer count. Hard‑capped ~25–27 tok/s on one card; the only jump is the rig.

### 122B‑A10B (qwen35moe HYBRID) — smartest tier, offloaded
Start from the 35B config below; offloaded single‑P100 ≈ **18 tok/s**. MTP via `--spec-type mtp` if the
GGUF retains `nextn` tensors (see [`MTP.md`](MTP.md)).

### 35B‑A3B (qwen35moe HYBRID, e.g. opus‑minime) — full‑GPU on one P100
```
-c 16384 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja --spec-type mtp:n_max=3,p_min=0.5 \
  --temp 0.6 --top-p 0.9 --top-k 40 --min-p 0.05 --repeat-penalty 1.08 --repeat-last-n 256 --presence-penalty 0.1
```
- ~45–48 tok/s with MTP n3 (+13–26% over no‑MTP), VRAM ~14.2/16 GB. **Budget context for the MTP context:**
  MTP allocates a separate speculative‑decode context + the FA/spec workspace scales with `n_ctx`;
  `-c 32768` OOMs on 16 GB, `-c 16384` fits. KV itself is cheap for A3B.

## Multi‑GPU notes
- **Layer‑split** crosses only ~4 KB/token (hidden state) → PCIe‑fine, robust without NVLink. Use it for
  models too big for one card. Avoid tensor/row split on PCIe/no‑NVLink (all‑reduce every layer).
- For a **mixed** P100 + 1080 Ti split with a low‑bit quant (Q2_K), **favor the 1080 Ti** (`-ts 8,16`) — its
  DP4A computes 2‑bit faster per layer despite less bandwidth. We measured 46 tok/s 1080Ti‑favored vs 40
  tok/s P100‑favored.
- For the future **multi‑P100 full‑GPU rig** (no offload), ik's `--split-mode graph` (graph‑parallel over
  PCIe) is the lever that unlocks real multi‑card decode concurrency; `SCHED_MAX_COPIES>=2` enables pipeline
  double‑buffering, but only on a **full‑GPU** model (it's gated off when `--n-cpu-moe`/`-ot` is present).

## TL;DR
- One P100: build F16/FORCE_DMMV, run Q3_K_M, expect ~55 tok/s (30B) / ~25 tok/s (offloaded 80B). Don't
  bother with bandwidth levers — you're dequant‑bound. Use MTP for np=1 turbo.
- P100 + 1080 Ti: stock sm_60;61 build, layer‑split, favor the 1080 Ti for low‑bit quants.
- Want more speed? Add cards. Tuning one card is a solved, capped problem.
