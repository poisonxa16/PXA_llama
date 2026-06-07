# MTP self‑speculation on Pascal

Multi‑Token Prediction (MTP) self‑speculation gives a clean, **lossless +21%** decode speedup on the P100
for qwen35moe models at np=1 — and it's a fork‑relevant feature because ik_llama has the MTP plumbing that
mainline historically drops on load.

## Proof (P100, Claude‑distilled Qwen3.6‑35B‑A3B‑MTP Q2_K, np=1, code prompt)

| config | tok/s | vs baseline | draft acceptance | SM%/mem% |
|---|---|---|---|---|
| baseline (no spec) | 43.3 | — | — | 93/15 |
| **MTP n_max=2** | **52.3** | **+21% (1.21×)** | 83% | 89/14 |
| MTP n_max=3 | 51.9 | +20% | 81% | 88/15 |

- **It works on Pascal**, and the overhead fear was unfounded: SM% went *down* (93→89). The batched verify
  amortizes the dequant bottleneck (see [`COMPUTE-BOUND-PASCAL.md`](COMPUTE-BOUND-PASCAL.md)) and does more
  useful work per cycle.
- **n_max=2 is the sweet spot** (n_max=3 slightly lower: more wasted drafts).
- Modest vs modern cards' 1.8–2.4× (Pascal has no tensor cores → batched verify less efficient) but real + free.

## How to use it

```
--spec-type mtp:n_max=2,p_min=0.5     # ik / pxa_llama
--spec-type draft-mtp                  # mainline (where supported)
```

## Constraints

- **np=1 only.** There is a hard guard: "MTP supports only n_parallel=1". MTP is inherently a
  low‑batch / single‑stream win — the speedup shrinks as a batch fills the GPU (compute‑bound at high batch),
  helping ~np2 and fading by np4. Run a **two‑tier** setup: a np=1 MTP‑turbo server for solo agentic/coding
  + a np=4 pxa_llama den for concurrent serving.
- **The GGUF must retain the `nextn` tensors.** The flag only engages MTP if they're present:
  ```
  grep -a 'blk.0.nextn' model.gguf   # present => MTP works
  ```
  - HAS the head: Qwen3.6‑35B‑A3B (huihui abliterated MTP, Claude‑distilled; llmfan "Native‑MTP‑Preserved"),
    Qwen3.6‑27B abliterated MTP — i.e. the fast tier, exactly where MTP matters.
  - STRIPPED in common GGUFs: Coder‑30B, Coder‑Next‑80B, Qwen3‑Next‑80B, some 122B builds.
- **Architecture support:** MTP works for `qwen3moe` / `qwen35moe` only. **qwen3next (Qwen3‑Next‑80B,
  Coder‑Next‑80B) is NOT supported** by the MTP path, and those GGUFs strip `nextn` anyway → the 80B is a
  pure (no‑MTP) smart tier.

## Grafting an MTP head onto a stripped model

For uncensored models whose GGUF stripped the head (e.g. some 122B builds), you can transplant an MTP head
from a donor with a pure GGUF tensor merge (CPU‑only, preserves quant + per‑row metadata):

```
convert.py merge base.gguf donor.gguf base-with-MTP.gguf
```

- Donors exist on HF (e.g. a Qwen3.5‑122B‑A10B MTP donor). Both base and donor must be the same arch
  (qwen35moe).
- **Caveat (test before trusting):** the donor head was trained on the base model's hidden states; grafted
  onto an abliterated variant the hidden states shift, so acceptance may be lower than a native head's ~83%.

## The pxa_llama angle

The np=1 limitation exists because "batching with MTP using a recurrent model is not optimized" — the *same*
multi‑seq recurrent‑state‑under‑concurrency problem that pxa_llama's v4 batched‑delta‑net fix already cracks.
So pxa_llama is uniquely positioned to explore **np>1 MTP** for hybrid Qwen3.x where stock can't. That's a
future lead, bounded by the fundamental low‑batch nature of speculation.
