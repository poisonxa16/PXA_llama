# Docker — run the prebuilt P100 server

The fastest way to try pxa_llama: a thin image around the prebuilt `llama-server` from
[Releases](https://github.com/poisonxa16/PXA_llama/releases/latest). **Tesla P100 (sm_60) only.**

## Build
```bash
docker build -t pxa_llama docker/
# pin a specific release instead of latest:
#   docker build --build-arg RELEASE=v0.1.0 -t pxa_llama docker/
```

## Run
```bash
docker run --rm --runtime=nvidia --gpus all -p 8088:8088 \
  -v /path/to/models:/models pxa_llama \
  -m /models/model.gguf -c 16384 -ngl 99 -np 4 -fa on \
  -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8088
```
The entrypoint is `run.sh` (sets `LD_LIBRARY_PATH` and execs `llama-server`), so everything after the image
name is passed straight to the server. See [`../docs/OLD-CARD-GUIDE.md`](../docs/OLD-CARD-GUIDE.md) for the
per-model flag sets.

## Verify concurrent-hybrid correctness
```bash
../benchmarks/concurrency-test.sh http://127.0.0.1:8088 6 chat   # -> CLEAN
```

## docker compose
```yaml
services:
  pxa_llama:
    build: ./docker
    runtime: nvidia
    ports: ["8088:8088"]
    volumes: ["/path/to/models:/models"]
    command: >
      -m /models/model.gguf -c 16384 -ngl 99 -np 4 -fa on
      -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8088
```

> Multi-card: a multi-**P100** rig can use this same sm_60 image as-is. A mixed **P100 + 1080 Ti** split
> needs a stock `sm_60;61` (no-F16) build — see [`../build/build-multicard.sh`](../build/build-multicard.sh)
> (and the separate multi-card release asset, when available).
