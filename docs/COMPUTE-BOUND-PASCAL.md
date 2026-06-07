# Pascal decode is dequant/compute‑bound, not bandwidth‑bound

A finding that reframes single‑card tuning on the P100 — and explains why so many "obvious" speed levers do
nothing.

## The assumption we falsified

For a long time we assumed the ~55 tok/s ceiling on a P100 (30B‑A3B Q3_K_M) was the **HBM2 bandwidth wall**
(~732 GB/s). Direct measurement proves that wrong.

## The measurement

`nvidia-smi dmon -s u` during sustained 30B‑A3B Q3_K_M decode:

| metric | value during decode |
|---|---|
| **SM (compute) utilization** | **95–99% (PEGGED)** |
| **Memory‑controller utilization** | **15–21% (mostly IDLE)** |

If decode were bandwidth‑bound, mem% would be ~90–100% and SM% lower. It is the **opposite**: the SMs are
maxed while the memory bus is ~84% idle.

## Diagnosis: dequant‑compute‑bound

The P100 (sm_60) has **no DP4A / int8 acceleration**. Unpacking each Q3_K superblock (scales/mins →
FP16 reconstruction) runs on the slow general ALU path. The cores spend ~96% of their time **dequantizing**;
the HBM2 bus sits at ~16% because the SMs can't consume data fast enough to saturate it.

## Why this explains every prior result

- IQK same‑size quant (IQ3_KS) = same speed → cost is dequant *compute*, not bytes.
- Spec‑decode for MoE = dead → the K draft tokens route to different experts → verify still dequant‑bound,
  zero amortization.
- KV‑type / context / `-ub` / `-ser` all neutral → none of them touch the dequant compute.
- Mainline slower → its kernels aren't better at Pascal dequant either.
- F16/FORCE_DMMV build helps (+19%) → it *does* touch the compute path (half2 HFMA2 instead of scalar).
- MTP helps (+21%, np=1) → batched verify does more useful work per cycle and *lowers* SM% (93→89).

## The opportunity (and the honest bound)

The bus being 84% idle means there is real headroom — a GP100‑tuned Q3_K dequant+matvec kernel (vectorized
128‑bit HBM2 loads, P100 2:1 FP16 half2 dequant arithmetic, overlap loads with compute) could roughly
double decode if dequant cost halves.

But after deeper analysis the per‑token wall is **fixed overhead** (attention + KV + ~48 kernel launches +
dequant) more than expert‑weight reads (~7% of decode wall‑time). So a custom dequant kernel attacks a
slice, not the whole. The biggest lossless lever is **MTP / EAGLE‑style self‑speculation** (amortizes the
fixed overhead over K tokens) — already shipped for qwen35moe. And the real multiplier remains a
**multi‑P100 full‑GPU rig** (more aggregate bandwidth + non‑contending slots).

## Takeaway for tuners

Stop reaching for bandwidth levers on one Pascal card — measure `dmon -s u` first. If SM% is pegged and
mem% is idle, you're dequant‑bound: only compute‑path changes (build flags, speculation, more cards) will
move the needle.
