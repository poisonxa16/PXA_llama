# Run the prebuilt pxa_llama llama-server (Tesla P100 / sm_60) in a container.
#
#   docker build -t pxa_llama docker/
#   docker run --rm --runtime=nvidia --gpus all -p 8088:8088 \
#     -v /path/to/models:/models pxa_llama \
#     -m /models/model.gguf -c 16384 -ngl 99 -np 4 -fa on \
#     -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8088
#
# P100 (sm_60) only. The CUDA 12.8 devel base provides glibc 2.38 / GLIBCXX 3.4.31 that the binary needs.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG RELEASE=latest
ARG ASSET=pxa_llama-sm60-p100-linux-x64.tar.gz
ARG REPO=poisonxa16/PXA_llama

RUN apt-get update -qq && apt-get install -y -qq curl ca-certificates jq \
 && rm -rf /var/lib/apt/lists/*

# Pull the release tarball (defaults to the latest release asset) and unpack it.
RUN set -e; \
    if [ "$RELEASE" = "latest" ]; then \
      URL="https://github.com/$REPO/releases/latest/download/$ASSET"; \
    else \
      URL="https://github.com/$REPO/releases/download/$RELEASE/$ASSET"; \
    fi; \
    echo "fetching $URL"; \
    mkdir -p /opt/pxa_llama; \
    curl -fsSL "$URL" | tar xz -C /opt/pxa_llama --strip-components=1

EXPOSE 8088
ENTRYPOINT ["/opt/pxa_llama/run.sh"]
CMD ["--help"]
