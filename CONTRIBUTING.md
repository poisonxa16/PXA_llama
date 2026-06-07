# Contributing to pxa_llama

pxa_llama is a small, focused fork: **our delta on top of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)**
(itself a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)). Contributions are welcome — please keep
that shape.

## Ground rules

1. **Engine changes are patches on top of ik_llama.cpp.** Don't vendor a whole modified tree. Update
   `patches/pxa_llama_v4_batched_delta_net.diff` (or add a new patch) and keep `CHANGELOG.md` in sync with the
   exact, file-level delta.
2. **Preserve upstream attribution.** Never remove copyright headers or the `ATTRIBUTION.md` / `LICENSE` notices.
   If a change really belongs upstream (kernels, quants, generic features), consider sending it to ik_llama.cpp
   or llama.cpp instead.
3. **Correctness gate is mandatory for engine changes.** Run, on a hybrid (qwen3next / qwen35moe) model at np>1:
   ```
   benchmarks/concurrency-test.sh <url> 6 chat   # must print: CLEAN
   ```
   A change that makes concurrent hybrid decoding dirty will not be merged.
4. **Be honest about numbers.** Report hardware, commands, and the regime. Decode on Pascal is
   dequant/compute-bound (see `docs/COMPUTE-BOUND-PASCAL.md`) — measure with `benchmarks/speed-test.sh`, don't
   assume. "It should be faster" is not a benchmark.
5. **Shell scripts** should pass `shellcheck --severity=error` (CI enforces it).

## Scope

This project is specifically about running modern hybrid/MoE LLMs **correctly and fast on old Pascal cards**
(Tesla P100 sm_60, GTX 1080 Ti sm_61). Great contributions: Pascal kernel/quant tuning, the multi-card path,
better build scripts, more correctness/benchmark coverage, decomposer improvements. Out of scope: the broader
home-server stack this fork was extracted from.

## DCO / license

By contributing you agree your work is released under the project's MIT license (see `LICENSE`).
