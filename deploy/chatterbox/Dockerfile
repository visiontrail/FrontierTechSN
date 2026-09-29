FROM python:3.11-slim-bookworm AS runtime_dependencies

ARG CHATTERBOX_SOURCE_REVISION=5de7a54aa4e5e2baadb0182dde554908b48b85c2
ARG PERTH_SOURCE_REVISION=ff1c8ac55a976971245cdd53c18d6131ca00d993
ARG DEBIAN_MIRROR=deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_XET=1 HF_HOME=/cache/huggingface \
    CHATTERBOX_DATA_DIR=/data CHATTERBOX_DEVICE=cpu
WORKDIR /app

RUN for apt_source in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
        if [ -f "$apt_source" ]; then \
            sed -i "s|http[s]*://deb.debian.org|https://${DEBIAN_MIRROR}|g" "$apt_source"; \
        fi; \
    done \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
        git build-essential libsndfile1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
# Optional reviewed wheel mirror. Keep these after the system packages so a
# slow-network deployment can reuse its completed APT layer.
ARG PIP_FIND_LINKS=""
ARG PIP_NO_INDEX=""
ARG PIP_TRUSTED_HOST=""
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r /app/requirements.txt

FROM runtime_dependencies AS final
# Optional SHA256-qualified mirrors of the exact pinned GitHub source archives.
# pip records their content hashes in direct_url metadata and the runtime freeze.
ARG CHATTERBOX_SOURCE_ARCHIVE=""
ARG PERTH_SOURCE_ARCHIVE=""
# On Linux aarch64, these PyPI Torch 2.6 wheels are CPU-only. Fail the build on
# accidental CUDA installation rather than taking another service's resources.
RUN python -c "import os,re; assert all(not os.getenv(k) or re.search(r'#sha256=[0-9a-f]{64}$', os.environ[k]) for k in ('CHATTERBOX_SOURCE_ARCHIVE','PERTH_SOURCE_ARCHIVE')), 'Source archive overrides require full SHA256 fragments'" \
    && python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" --no-deps \
        "${PERTH_SOURCE_ARCHIVE:-git+https://github.com/resemble-ai/Perth.git@${PERTH_SOURCE_REVISION}}" \
        "${CHATTERBOX_SOURCE_ARCHIVE:-git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_SOURCE_REVISION}}" \
    && python -c "import torch, torchaudio; from chatterbox.mtl_tts import ChatterboxMultilingualTTS; import perth; assert torch.__version__.startswith('2.6.0'); assert torch.version.cuda is None; assert perth.PerthImplicitWatermarker is not None" \
    && python -m pip freeze > /app/runtime-requirements.txt \
    && apt-get purge -y --auto-remove git build-essential \
    && rm -rf /root/.cache/pip
COPY service.py prefetch_pkuseg.py /app/
COPY space_alignment.py apply_space_alignment.py /app/
RUN python apply_space_alignment.py && python space_alignment.py
ENV PKUSEG_HOME=/cache/pkuseg
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
