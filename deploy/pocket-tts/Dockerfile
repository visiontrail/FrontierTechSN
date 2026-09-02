FROM ghcr.io/astral-sh/uv:debian@sha256:b2b767dc8bbe5b9f813ca4958a43cc8421ca8eab6c954c74e5b1e641fea7a6ad

ARG POCKET_TTS_CPU_TORCH_URL="https://download-r2.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=269b10c34430aa8e9643dbe035dc525c4a9b1d671cd3dbc8ecbcaed280ae322d"

ENV UV_LINK_MODE=copy

WORKDIR /app
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./README.md .
COPY ./.python-version .
COPY ./pocket_tts ./pocket_tts

# The upstream Dockerfile uses plain `uv run`, which installs both development
# tooling and PyPI's CUDA-enabled Linux torch graph. Pocket serves on CPU by
# default, and upstream explicitly supports torch >=2.5 (while warning that 2.4
# produces incorrect audio). Install the official, hash-pinned ARM64 CPU wheel,
# retain the lock's exact versions for every other runtime package, then install
# only the local model package itself.
RUN uv python install 3.10 \
    && uv venv --python 3.10 \
    && uv export --frozen --no-dev --no-hashes --no-header --no-emit-project \
       | grep -Ev '^(torch==|triton==|nvidia-|cuda-)' > /tmp/runtime-requirements.txt \
    && uv pip install "${POCKET_TTS_CPU_TORCH_URL}" \
    && uv pip install --index-url https://pypi.org/simple -r /tmp/runtime-requirements.txt \
    && uv pip install --no-deps . \
    && uv run --no-sync python -c "import torch; assert torch.__version__.startswith('2.5.1'); assert torch.version.cuda is None" \
    && uv run --no-sync pocket-tts --help \
    && rm /tmp/runtime-requirements.txt

ENTRYPOINT ["uv", "run", "--no-sync", "pocket-tts"]
