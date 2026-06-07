# Benchmark harnesses

Two small, dependency‑light scripts (need `curl` + `jq`). Point them at a running `llama-server`.

## `concurrency-test.sh` — the correctness moat
Fires K concurrent requests, each carrying a unique codeword, and checks every reply contains **only its
own** codeword. Recurrent‑state bleed shows up as the wrong codeword/garbage in a reply.

```bash
./concurrency-test.sh http://127.0.0.1:8088 6 chat
# verdict: CLEAN  (exit 0)  |  DIRTY (exit 1)
```
- Run it against a **stock ik_llama** hybrid build → DIRTY at K≥4.
- Run it against **pxa_llama** (v4 fix) → CLEAN at K=4 and K=6.
- `mode` = `chat` (`/v1/chat/completions`, default) or `completion` (`/completion`).

## `speed-test.sh` — decode throughput
Single‑stream and aggregate‑concurrent tok/s, using llama‑server's own timing fields. Use it to compare
stock vs the F16 build, or to characterize the batching curve.

```bash
./speed-test.sh http://127.0.0.1:8088 4 200      # base_url  concurrency  n_predict
```

Decode on Pascal is dequant/compute‑bound (see [`../docs/COMPUTE-BOUND-PASCAL.md`](../docs/COMPUTE-BOUND-PASCAL.md)),
so don't expect KV‑type/context/`-ser` to move single‑stream — the RATIO under concurrency is the
interesting number on a multi‑slot rig.
