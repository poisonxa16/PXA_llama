<!-- Thanks for contributing to pxa_llama. -->

## What & why


## How tested
<!-- For anything touching the engine, the correctness gate is mandatory: -->
- [ ] `benchmarks/concurrency-test.sh <url> 6 chat` → CLEAN (on a hybrid model, np>1)
- [ ] `benchmarks/speed-test.sh <url>` (note any tok/s change)
- [ ] Hardware: <!-- GPU(s), driver/CUDA, model+quant -->

## Notes
- [ ] If this changes the engine, the change is a **patch on top of ik_llama.cpp** (keep `patches/` and `CHANGELOG.md` in sync).
- [ ] Upstream attribution preserved (no removed copyright/notices).
