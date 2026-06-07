# pxa_llama

[![ci](https://github.com/poisonxa16/PXA_llama/actions/workflows/ci.yml/badge.svg)](https://github.com/poisonxa16/PXA_llama/actions/workflows/ci.yml)
[![latest release](https://img.shields.io/github/v/release/poisonxa16/PXA_llama?sort=semver)](https://github.com/poisonxa16/PXA_llama/releases/latest)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GPU: Tesla P100 / 1080 Ti](https://img.shields.io/badge/GPU-Tesla%20P100%20%2F%20GTX%201080%20Ti-76b900.svg)](docs/OLD-CARD-GUIDE.md)
[![based on ik_llama.cpp](https://img.shields.io/badge/fork%20of-ik__llama.cpp-555.svg)](https://github.com/ikawrakow/ik_llama.cpp)

**Run modern hybrid / MoE LLMs *correctly and fast* on cheap, old Tesla P100 / GTX 1080 Ti cards.**

pxa_llama is a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (itself a fork of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)) that adds the pieces you need to run
**concurrent hybrid Gated‑DeltaNet MoE models** (Qwen3‑Next, Qwen3.5‑MoE / "qwen35moe") on Pascal‑era
datacenter GPUs that the upstream projects don't even target — and to run them *correctly* under
concurrency, where stock ik_llama silently corrupts output.

It exists because a ~$150 eBay Tesla P100 has something modern budget cards don't: **fast 2:1 FP16 and
732 GB/s HBM2**. With the right build and one real bug fix, it will serve a concurrent 122B‑A10B hybrid
MoE without cross‑conversation bleed.

**The pitch in one line:** the kernel fix makes np>1 hybrid decoding correct, and a stack of
Pascal‑specific enhancements — F16/FORCE_DMMV build tuning, MTP self‑speculation, DP4A‑aware multi‑card
splitting, and a built‑in size‑gated fan‑out decomposer in the server — squeeze the **fastest tok/s these
old cards can give, on single cards and multi‑card rigs alike.**

> This is an honest fork. See [`ATTRIBUTION.md`](ATTRIBUTION.md) and [`LICENSE`](LICENSE). Everything here
> is **our delta on top of ik_llama.cpp / llama.cpp**, not a from‑scratch engine. The unique value is the
> patches in [`patches/`](patches/), the Pascal build tuning, and the measured results in
> [`BENCHMARKS.md`](BENCHMARKS.md).

---

## The headline: concurrent hybrid decoding that isn't garbage

Stock ik_llama.cpp **corrupts hybrid / recurrent‑state models** (qwen3next Gated‑DeltaNet, qwen35moe) at
`np>=3`: concurrent slots bleed each other's recurrent state → garbage / cross‑conversation output.
pxa_llama's **v4 batched‑delta‑net fix** makes `np>1` clean.

Distinct‑codeword cross‑bleed test (each concurrent slot must return **only its own** codeword):

| Engine | np=1 | np=2 | np=4 | np=6 |
|---|---|---|---|---|
| stock ik_llama (hybrid) | clean | clean | **CORRUPT (cross‑bleed)** | **CORRUPT** |
| **pxa_llama (v4 fix)** | clean | clean | **CLEAN** | **CLEAN** |

*Nobody else runs concurrent hybrid MoE correctly on these cards.* Reproduce it yourself with
[`benchmarks/concurrency-test.sh`](benchmarks/concurrency-test.sh) — see [`BENCHMARKS.md`](BENCHMARKS.md).

---

## What's unique (our contributions)

1. **Clean concurrent hybrid (np>1) Gated‑DeltaNet** — the flagship. A batched, multi‑seq delta‑net path
   that replaces ik's per‑token loop (which trips the ggml graph allocator into reusing live recurrent
   scratch across concurrent sequences). Full root‑cause writeup:
   [`docs/HYBRID-CONCURRENCY-BUG.md`](docs/HYBRID-CONCURRENCY-BUG.md). Patch:
   [`patches/pxa_llama_v4_batched_delta_net.diff`](patches/pxa_llama_v4_batched_delta_net.diff).
2. **Pascal / old‑card build tuning** — an sm_60 build with `GGML_CUDA_F16=ON` + `GGML_CUDA_FORCE_DMMV=ON`
   (+`MMV_Y=2`) that leverages the P100's 2:1 FP16 path for dequant + matmul: **~+19% over stock**, build‑time.
   Stock llama.cpp / ik images are sm_61+ only and don't target the P100 at all.
3. **The DP4A / FP16 economics for old cards, documented with data** — the P100 (sm_60) has fast FP16 but
   **no int8/DP4A**; the 1080 Ti (sm_61) **has DP4A** so it's faster for low‑bit quants (Q2_K) despite less
   bandwidth → favor the 1080 Ti in a mixed layer‑split. Decode is dequant/compute‑bound on Pascal, not
   bandwidth‑bound (measured). Full guide: [`docs/OLD-CARD-GUIDE.md`](docs/OLD-CARD-GUIDE.md).
4. **MTP (Multi‑Token Prediction) self‑speculation on Pascal** — `--spec-type mtp` gives a clean, lossless
   **+21%** on qwen35moe at np=1, proven on a P100. [`docs/MTP.md`](docs/MTP.md).
5. **Built‑in fan‑out decomposer — compiled into the server, toggled on/off by a flag.** A 68 MB bilingual
   dependency cross‑encoder (`nextn`‑style head, embedded in the llama.cpp binary as `llama-decompose-server`,
   pure ggml, no Python/ONNX at runtime) that auto‑decomposes a prompt into a dependency DAG and **fans the
   independent sub‑tasks out across the cards/slots**, with a **built‑in size gate** so it only fans out when
   the work is substantial enough to pay. An auto, gated, validated Skeleton‑of‑Thought, *inside the engine*.
   **99/100** on a held‑out gauntlet, **0** dangerous false‑parallels, ~10 ms/decision on CPU. Honest about
   where it pays. [`decomp-router/`](decomp-router/).

---

## Results at a glance (full methodology in [`BENCHMARKS.md`](BENCHMARKS.md))

| What | Number | Card(s) |
|---|---|---|
| Concurrent hybrid np=4/6 | **CLEAN** (stock = corrupt) | 1× P100 |
| Pascal F16/FORCE_DMMV build | **~+19%** decode vs stock | 1× P100 |
| 30B‑A3B Q3_K_M, full‑GPU | **~55 tok/s** single‑stream, ~83 agg @ np4 | 1× P100 |
| 80B Coder‑Next Q3, offloaded | ~25.7 tok/s (PCIe‑capped) | 1× P100 |
| 122B‑A10B Q3, offloaded | ~18 tok/s | 1× P100 |
| MTP self‑spec (qwen35moe, np=1) | **+21%** lossless | 1× P100 |
| 35B‑A3B Q2_K, full‑GPU layer‑split | **~46 tok/s**, 2.18× batching @ N≈4 | P100 + 1080 Ti |
| Decomposer accuracy / latency | **99/100**, 0 dangerous, ~10 ms | CPU |
| Fan‑out on substantial 3‑way work | **1.73×** | P100 + 1080 Ti |

---

## Quick start (10 minutes on a P100)

### Option A — grab the prebuilt P100 binary (fastest)
A ready-to-run `llama-server` for the **Tesla P100 (sm_60)** is attached to the
[**latest release**](https://github.com/poisonxa16/PXA_llama/releases/latest) (the v4 fix + the
F16/FORCE_DMMV speed build):
```bash
# download pxa_llama-sm60-p100-linux-x64.tar.gz from the Releases page, then:
tar xzf pxa_llama-sm60-p100-linux-x64.tar.gz && cd pxa_llama-sm60-p100-linux-x64
./run.sh -m model.gguf -c 16384 -ngl 99 -np 4 -fa on -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8088
```
> P100 (sm_60) only — **not** for a 1080 Ti. Needs glibc ≥ 2.38 / a CUDA 12.x runtime; easiest is to run
> inside `nvidia/cuda:12.8.1-devel-ubuntu24.04` with `--runtime=nvidia`. `run.sh` just sets
> `LD_LIBRARY_PATH=./lib` and execs `./llama-server`.

### Option B — build from source

```bash
# 1. Get the upstream source (pxa_llama is a patch set on top of ik_llama.cpp)
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
git checkout 1520eda   # the base this patch was cut against; or apply with -3 / fuzz on a newer HEAD

# 2. Apply the pxa_llama concurrent-hybrid fix
git apply --3way /path/to/pxa_llama/patches/pxa_llama_v4_batched_delta_net.diff

# 3. Build the Pascal (sm_60) speed binary — runs in a CUDA devel container
/path/to/pxa_llama/build/build-sm60.sh    # see build/README.md

# 4. Serve a hybrid MoE model (offloaded 122B-A10B shown; 16GB card)
#    (run inside nvidia/cuda:12.8.1-devel-ubuntu24.04 with LD_LIBRARY_PATH set — see launchers/)
./llama-server -m qwen3.5-122B-A10B-Q3_K_M.gguf \
  -c 16384 -ngl 99 --n-cpu-moe 48 -np 4 -fa on -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8088

# 5. Prove concurrent decoding is clean (THIS is the moat)
/path/to/pxa_llama/benchmarks/concurrency-test.sh http://127.0.0.1:8088 6 chat
#   -> verdict: CLEAN   (run the same against a stock ik_llama build to see CORRUPT at K>=4)
```

Ready‑to‑use launchers (model + flags + the docker invocation with the scattered‑lib `LD_LIBRARY_PATH`)
live in [`launchers/`](launchers/). The per‑model "max settings" are in
[`docs/OLD-CARD-GUIDE.md`](docs/OLD-CARD-GUIDE.md).

---

## Honest limits (read this — the regime is the point)

- **Single‑card decode is dequant/compute‑bound on Pascal, not bandwidth‑bound.** We measured the SMs pegged
  at 95–99% while the HBM2 bus sat at ~16% during decode (no DP4A → Q3_K superblock unpacking runs on the
  slow ALU path). So bandwidth‑oriented levers (KV‑type, context size, `-ser`, IQK same‑size quants,
  spec‑decode for MoE) are **neutral** on one offloaded card — we tested them and say so. See
  [`docs/COMPUTE-BOUND-PASCAL.md`](docs/COMPUTE-BOUND-PASCAL.md).
- **The built‑in fan‑out only wins on under‑utilized rigs** (which is why it's a toggle, off by default). It's
  a tuned, gated implementation of a known idea — **Skeleton‑of‑Thought** (Ning et al., arXiv
  [2307.15337](https://arxiv.org/abs/2307.15337), up to 2.39×). On a busy multi‑tenant server (already
  batch‑saturated) it doesn't help; on short prompts it loses (N× prefill + token bloat), which is exactly
  why there's a **size gate**. Our contribution is the engineering — putting it *in the server* — for the
  homelab/personal‑agent niche, not a universal free lunch.
- **The real speed jump is more cards, not more tuning.** One P100 is ~55 tok/s (30B‑A3B) / ~25 tok/s
  (offloaded 80B); the only path past that is a full‑GPU multi‑P100 rig (no PCIe offload wall).
- **MTP is a np=1 / low‑batch win** and only engages on GGUFs that retain the `nextn` tensors.

---

## Repo layout

```
README.md                  – this file
LICENSE                    – MIT (inherited from llama.cpp / ik_llama.cpp)
ATTRIBUTION.md             – provenance: what is upstream, what is ours
CHANGELOG.md               – the exact deltas vs ik_llama.cpp
BENCHMARKS.md              – every number, with hardware + commands + methodology
patches/                   – the v4 concurrent-hybrid fix + how to apply + evolution log
docs/
  HYBRID-CONCURRENCY-BUG.md  – root cause of the np>1 corruption + the fix (the moat)
  OLD-CARD-GUIDE.md          – Pascal FP16/DP4A/quant tuning + per-model max settings
  COMPUTE-BOUND-PASCAL.md    – the dequant-bound measurement + what it means
  MTP.md                     – MTP self-speculation on Pascal
build/                     – sm_60 (P100) and multi-card (sm_60;61) build scripts + notes
benchmarks/                – the concurrency-correctness + speed harnesses
launchers/                 – ready-to-run llama-server invocations (docker + flags)
decomp-router/             – the built-in server fan-out decomposer (in-engine code + eval gauntlets)
```

## Acknowledgements

Built on the excellent work of **Kawrakow** ([ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp))
and the **ggml-org / llama.cpp** community. pxa_llama would not exist without them. All upstream code
remains under its original MIT license.
