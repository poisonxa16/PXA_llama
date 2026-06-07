# Decomposition router (companion)

An auto, gated, validated **Skeleton‑of‑Thought** for idle multi‑GPU rigs: a purpose‑built ultra‑light model
decomposes a prompt into a dependency DAG and the engine fans out the **independent** sub‑tasks across np
slots in parallel, then merges. The point is to cut wall‑clock latency for genuinely multi‑part prompts on a
rig that has idle slots — *without ever false‑parallelizing a dependent step* (which would corrupt output).

> **Honest framing first.** This is a tailored, gated implementation of a **known idea** — Skeleton‑of‑Thought
> (Ning et al., arXiv [2307.15337](https://arxiv.org/abs/2307.15337), up to 2.39×). It is **not** a universal
> free lunch. See "Where it pays / where it doesn't" below. Our contribution is the engineering: a sub‑10ms
> in‑engine decomposer (instead of using the big LLM to build the skeleton), a dependency‑DAG safety layer, a
> size gate, and bilingual validation.

## opus‑microme — the decomposer model

- **Backbone:** `Geotrend/distilbert-base-en-es-cased` (vocab pruned to English + Spanish), fine‑tuned as a
  **dependency cross‑encoder**: input `(prompt, clauseA, clauseB)` → `P(B depends on A)`.
- **Pipeline:** segmenter splits the prompt → every ordered clause‑pair is classified → a DAG is built →
  fan‑out iff the DAG has ≥2 concurrent nodes. Loss is weighted toward *dependent* so it **fails safe**
  (serialize) rather than dangerous (parallelize).
- **Accuracy (held‑out gauntlets, never trained on):** **99/100** edge cases, **0 dangerous false‑parallels**
  — across comma‑traps ("Tokyo, Japan"), quote‑traps, binomials ("pros and cons"), possessive anaphora
  (EN "its" / ES "su"), find‑then‑fix chains, and Spanish hidden dependencies. (1 conservative miss.)
- **Size & speed:** 68 MB int8 ONNX, **~10.9 ms/pair on CPU** — runs on idle cores, never the GPUs.

## In‑engine C++ implementation — `decompose-server.cpp`

The 68 MB model converted to GGUF (BERT body) + a 2‑layer head, embedded in a llama.cpp engine binary
(segment + ggml embed + head matmul + DAG, **no ONNX/Python at runtime**).
- Validated vs GOLD: **99/100, 0 dangerous** — bit‑identical to the ONNX model, pure ggml.
- Latency: **9 ms** atomic, **65–80 ms** multi‑clause (CPU, batchable).
- **Size gate:** emits `recommend` (≥2 independent *substantial* sub‑tasks) separate from the structural
  `fanout` → only fan out when it actually pays.

## Where it pays / where it doesn't (measured)

| Workload | Speedup |
|---|---|
| Substantial, 3 independent tasks (~150‑word each) | **1.73×** ✓ (≈80% of the 2‑card batching ceiling) |
| Substantial, 4 tasks | 1.30× |
| Short conversational (joke / movie / book) | **0.2–0.5× ✗** (gated out by `recommend`) |
| 5–6 tasks | noisy / regresses (token imbalance) |

- The win exists **only on under‑utilized rigs** (idle slots). A busy multi‑tenant server is already
  batch‑saturated → no help.
- Short prompts **lose** (N× prefill + standalone sub‑tasks bloat tokens) → that's why the **size gate** holds
  them linear.
- The safety property (never false‑parallelize a dependent step) holds **everywhere**, regardless of regime.

## Files
- `decompose-server.cpp` — the in‑engine C++/ggml decomposer (the production artifact).
- `testset.json` / `testset2.json` / `testset3.json` — the held‑out gauntlets (EN, hard EN, ES+EN).

The training corpus, ONNX export, and Python eval harness are large and live in the project's R&D tree; the
model is reproducible from a seeded generator → distilbert fine‑tune → ONNX/GGUF export. The two artifacts
that matter for a release — the in‑engine decoder and the validation gauntlets — are here.
