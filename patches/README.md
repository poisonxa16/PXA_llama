# Patches

## `pxa_llama_v4_batched_delta_net.diff` — the concurrent‑hybrid fix

The headline contribution: makes hybrid Gated‑DeltaNet MoE models (qwen3next / qwen35moe) decode **correctly
at np>1** on ik_llama.cpp. Stock ik corrupts them at np≥3 (cross‑conversation bleed). Full root‑cause
writeup: [`../docs/HYBRID-CONCURRENCY-BUG.md`](../docs/HYBRID-CONCURRENCY-BUG.md).

**What it does:** replaces ik's per‑token delta‑net loop (which trips the ggml graph allocator into reusing
live recurrent scratch across concurrent sequences) with **one batched, multi‑seq delta‑net call**,
mirroring ik's own concurrency‑clean Mamba path. 387‑line diff, localized to:
- `src/llama-delta-net.cpp` (`build_qkv`, `build_beta_gate`, `build_layer_attn_linear`)
- `src/llama.cpp` (per‑step routing‑tensor fill)

**Apply:**
```bash
cd ik_llama.cpp
git checkout 1520eda
git apply --3way pxa_llama_v4_batched_delta_net.diff
# or, on a different HEAD:
patch -p1 --fuzz=3 < pxa_llama_v4_batched_delta_net.diff
```

**Verify after building:**
```bash
../benchmarks/concurrency-test.sh http://127.0.0.1:8088 6 chat   # -> CLEAN
```

## Evolution (for the curious)

The fix took five iterations as we localized the true root cause. Each banked attempt changed the failure
mode — that's how we ruled causes in/out:

| version | approach | result |
|---|---|---|
| v0 | never reuse graph for recurrent | clean but 0.69 tok/s (unusable) — proved graph‑reuse staleness was *a* cause |
| v1 | seq‑signature reuse key | np=1 fast again; np>1 still dirty (deeper) |
| v2 | runtime per‑seq get_rows/set_rows gather | np=1 clean; np>1 dirty (per‑token in‑place aliasing) |
| v3 / v3b / v3c | batched / deferred‑scatter attempts | v3b clean to np=2, dirty at np≥3 (allocator hazard pinned) |
| **v4** | single batched multi‑seq op (no per‑token loop) | **clean to np≥6 — shipped** |

The takeaway: the per‑token‑**loop structure** was the bug (it created N interleaved subgraphs aliasing one
persistent recurrent buffer); no read/write‑mechanism tweak could fix it — only collapsing to one batched op
did. See the changelog and the bug doc for the gory details.
