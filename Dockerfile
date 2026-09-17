FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /opt/inspark_marlin
COPY . .
RUN bash scripts/bootstrap.sh
ENTRYPOINT ["bash", "scripts/run.sh"]
CMD ["-m", "acc_infer_clear.cli", "--help"]

