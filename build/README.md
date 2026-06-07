# Building pxa_llama

pxa_llama is a patch set on top of ik_llama.cpp. Build = clone upstream → apply the patch → compile with
the right arch flags for your card(s).

## 1. Get the source + apply the fix
```bash
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
git checkout 1520eda                                  # base the patch was cut against
git apply --3way ../pxa_llama/patches/pxa_llama_v4_batched_delta_net.diff
```
(On a newer ik HEAD, `git apply --3way` / `patch -p1 --fuzz=3` usually still lands — the change is localized
to `src/llama-delta-net.cpp` + a small bit of `src/llama.cpp`. See `patches/README.md`.)

## 2. Build for your hardware

| Your setup | Script | Flags |
|---|---|---|
| **Single Tesla P100** | `build-sm60.sh` | sm_60, **F16 + FORCE_DMMV + MMV_Y2** (~+19%) |
| **P100 + GTX 1080 Ti** (layer-split) | `build-multicard.sh` | sm_60;61, **stock (no F16)** |

```bash
../pxa_llama/build/build-sm60.sh /path/to/ik_llama.cpp
```

Both scripts build inside `nvidia/cuda:12.8.1-devel-ubuntu24.04` (the binary needs glibc 2.38 /
GLIBCXX 3.4.31 — build it where you'll run it, or in a matching runtime).

## 3. Important gotchas
- **Never put the F16 (`build-sm60`) binary on a 1080 Ti** — `GGML_CUDA_F16=ON` is catastrophic on 1:64 FP16.
- The ik libraries are **scattered** across the build tree. Run with:
  `LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd`
- Run the container with `--runtime=nvidia` and mount the build dir read-only at `/build`.
- A *failed* recompile preserves the prior binary — safe to iterate.

See `../launchers/` for complete, ready-to-run server invocations and `../docs/OLD-CARD-GUIDE.md` for the
per-model max settings.
