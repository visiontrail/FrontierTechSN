FROM python:3.10-slim-bookworm@sha256:68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a

ARG POCKET_TTS_CPU_TORCH_URL="https://download-r2.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=269b10c34430aa8e9643dbe035dc525c4a9b1d671cd3dbc8ecbcaed280ae322d"

ENV PYTHONPATH=/app

WORKDIR /app
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./README.md .
COPY ./.python-version .
COPY ./pocket_tts ./pocket_tts

# The upstream Dockerfile installs development tooling and PyPI's CUDA-enabled
# Linux torch graph. These versions are the non-CUDA, non-dev result of its
# exact uv.lock at adde0654. --no-deps makes the list closed and prevents pip
# from silently resolving a different graph. Upstream requires torch >=2.5 and
# warns that 2.4 produces incorrect audio, so the ARM64 CPU wheel is hash-pinned.
RUN python -m pip install --no-cache-dir --no-deps \
        "${POCKET_TTS_CPU_TORCH_URL}" \
        annotated-doc==0.0.5 \
        annotated-types==0.8.0 \
        anyio==4.14.2 \
        certifi==2026.7.22 \
        charset-normalizer==3.5.1 \
        click==8.4.2 \
        "colorama==0.4.6 ; sys_platform == 'win32'" \
        einops==0.8.2 \
        "exceptiongroup==1.3.1 ; python_full_version < '3.11'" \
        fastapi==0.141.1 \
        filelock==3.32.3 \
        fsspec==2026.7.0 \
        h11==0.16.0 \
        "hf-xet==1.6.0 ; platform_machine == 'AMD64' or platform_machine == 'aarch64' or platform_machine == 'amd64' or platform_machine == 'arm64' or platform_machine == 'x86_64'" \
        httpcore==1.0.9 \
        httpx==0.28.1 \
        huggingface-hub==1.28.0 \
        idna==3.19 \
        jinja2==3.1.6 \
        markdown-it-py==4.2.0 \
        markupsafe==3.0.3 \
        mdurl==0.1.2 \
        mpmath==1.3.0 \
        "networkx==3.4.2 ; python_full_version < '3.11'" \
        "networkx==3.6.1 ; python_full_version >= '3.11'" \
        "numpy==2.2.6 ; python_full_version < '3.11'" \
        "numpy==2.4.6 ; python_full_version == '3.11.*'" \
        "numpy==2.5.2 ; python_full_version >= '3.12'" \
        packaging==26.3 \
        pydantic==2.13.4 \
        pydantic-core==2.46.4 \
        pygments==2.21.0 \
        python-multipart==0.0.32 \
        pyyaml==6.0.3 \
        requests==2.34.2 \
        rich==15.0.0 \
        safetensors==0.8.0 \
        "scipy==1.15.3 ; python_full_version < '3.11'" \
        "scipy==1.17.1 ; python_full_version == '3.11.*'" \
        "scipy==1.18.0 ; python_full_version >= '3.12'" \
        sentencepiece==0.2.2 \
        setuptools==81.0.0 \
        shellingham==1.5.4 \
        starlette==1.6.0 \
        sympy==1.14.0 \
        tqdm==4.70.0 \
        typer==0.27.1 \
        typing-extensions==4.16.0 \
        typing-inspection==0.4.4 \
        urllib3==2.7.0 \
        uvicorn==0.52.4 \
    && python -c "import fastapi, pocket_tts, torch; assert torch.__version__.startswith('2.5.1'); assert torch.version.cuda is None" \
    && python -m pocket_tts --help

ENTRYPOINT ["python", "-m", "pocket_tts"]
