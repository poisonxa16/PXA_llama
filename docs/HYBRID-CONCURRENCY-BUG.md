# The np>1 hybrid concurrency bug — root cause & fix

This is the flagship contribution. If you only read one doc, read this.

## Symptom

Stock ik_llama.cpp, serving a **hybrid / recurrent‑state model** (qwen3next Gated‑DeltaNet — Coder‑Next‑80B,
Qwen3‑Next‑80B; or qwen35moe — Qwen3.5‑35B/122B) with `-np >= 3`, produces **garbage or
cross‑conversation output** under concurrent requests: one slot's reply contains another slot's content.

- `np=1`: clean.
- `np=2`: clean.
- `np>=3`: **dirty** — and raising `np` makes it worse.
- `cache_prompt:false` reduces but does **not** fix it.
- Non‑hybrid `qwen3moe` (e.g. 30B‑A3B, no recurrent state) is **clean at np=4** on stock ik — so the
  attention / MoE / KV concurrency is fine. The bug is in the **delta‑net (recurrent) path only**.

## Diagnosis (the journey, abbreviated)

We chased three distinct recurrent‑state bugs before finding the real one. Each *changed* the failure mode
but didn't eliminate it — which is how we eventually localized the true cause:

1. **Stale graph reuse.** ik routes a token to its recurrent state row via a **build‑time C++ scalar
   offset** baked into the graph (`state_seq_id_local * row_size`), while the runtime routing tensor
   `inp_s_seq_qnext` was hardcoded to 0. Graph reuse is on by default and its reuse key didn't reflect the
   active‑seq set, so a graph built for step N's seq→row mapping got reused at step N+1 after a slot
   started/finished/reordered → wrong row. *Fixing reuse alone (v0/v1) didn't fix np>1.*
2. **Build‑time state offset / no runtime gather.** Ported a runtime `get_rows`/`set_rows` gather from ik's
   own Mamba path (v2). np=1 clean, np>1 still dirty.
3. **Per‑token in‑place aliasing.** The mixed‑seq decode path builds **N independent per‑token subgraphs**,
   each issuing `get_rows`/`set_rows` against the *same* persistent recurrent pool `s_l[il]`.

### The real root cause
**A ggml graph‑allocator hazard created by the per‑token‑loop structure itself.**
`ggml-alloc.c` frees and reuses buffer offsets by topological refcount. With **N ≥ 3 interleaved per‑token
subgraphs** all reading/writing the one persistent recurrent buffer, a still‑live recurrent scratch tensor
gets its offset reused by another token's subgraph → cross‑sequence bleed. The cliff is exactly at np=2
clean / np≥3 dirty, matching the allocator's reuse pressure. No read/write‑*mechanism* tweak can fix it,
because the **loop structure** is the bug. (Corroboration: a single‑snapshot variant made even np=2 dirty —
the shared tensor was aliased by the N subgraphs.)

This also reconciles with the public ik issue about mixed‑seq hybrid decode falling back to slow
single‑token chunking: same root cause; the chunking guard catches some paths, the reuse path is unguarded.

## The fix (`PXA_LLAMA_FIX_v4`)

**Replace the per‑token loop with one batched, multi‑seq delta‑net call** — exactly how ik's *own* Mamba
path (`src/graphs/build_mamba.cpp`) does it, which is concurrency‑clean:

- One `ggml_ssm_conv` + one delta‑net pass over **all** tokens in the ubatch.
- A `[n_kv, n_tokens]` **sequence map** routes each token to its state row *inside* the kernel (built like
  `build_inp_s_seq` / the Mamba host‑fill), plus a `state_row_idx` gather and a `state_mask`.
- **One contiguous write‑back** instead of N in‑place scatters.

ik's delta‑net CUDA kernel and the `ssm_conv` multi‑seq‑unique path **already support `n_seqs>1`** — only
the graph *builder* was looping. So the fix is a graph‑construction change, not a kernel rewrite, and it
preserves all of ik's IQK speed.

### Files changed
- `src/llama-delta-net.cpp`
  - `build_qkv(...)` signature: `state_seq_id_local` (scalar) → `state_row_idx`, `conv_seq_map`,
    `state_mask`, `n_seqs_in`. Batched gather of the recurrent state (`[state_dim, n_seqs]`), batched
    conv with an identity seq‑map in token order, batched write‑back via one `set_rows`.
  - `build_beta_gate(...)`: reshape beta/gate on `(n_seq_tokens, n_seqs)` so the per‑seq batch dim flows
    into `build_fused_delta_net`.
  - `build_layer_attn_linear` mixed path: the per‑token loop → one batched core call (`n_seqs = n_tok`,
    `n_seq_tokens = 1`), `inp_out_ids` applied once after.
- `src/llama.cpp`: fill the per‑step routing tensors (state‑row map, conv seq map) with the real per‑seq
  indices instead of hardcoding 0.

The full 387‑line diff: [`../patches/pxa_llama_v4_batched_delta_net.diff`](../patches/pxa_llama_v4_batched_delta_net.diff).

## Verification

`benchmarks/concurrency-test.sh` fires K concurrent requests, each carrying a unique codeword, and fails on
any cross‑contamination or garbage:

```
benchmarks/concurrency-test.sh http://127.0.0.1:8088 4 chat   # CLEAN
benchmarks/concurrency-test.sh http://127.0.0.1:8088 6 chat   # CLEAN
```

- pxa_llama: **CLEAN at N=4 and N=6** (distinct codewords ZULU111/MANGO222/… all unique).
- stock ik_llama on the same hybrid model: **CORRUPT at N≥4**.
- Concurrent tool‑call gauntlet: 7/7 correct, no cross‑talk, no malformed calls.
- np=1 single‑stream speed unchanged.

## Why this matters

Hybrid Gated‑DeltaNet MoEs (Qwen3‑Next, Qwen3.5‑MoE) are some of the best uncensored local models you can
run, and their low active‑parameter MoE design is ideal for bandwidth‑limited old cards. But a serving
engine that corrupts them under concurrency is useless for any multi‑slot / multi‑agent / multi‑user setup.
pxa_llama is, as far as we know, the only build that serves them **correctly at np>1 on these cards.**
