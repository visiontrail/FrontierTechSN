"""Isolated, authenticated Chatterbox V2 service with durable idempotent jobs.

The HTTP process never imports Torch. One killable child owns the model and
global RNG; requests run serially because Chatterbox mutates voice conditioning.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import multiprocessing
import os
import queue
import sqlite3
import sys
import threading
import time
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from prefetch_reference import VOICE_ID as SPACE_EN_VOICE, reference_source, verify_reference

SOURCE_REVISION = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
PERTH_REVISION = "ff1c8ac55a976971245cdd53c18d6131ca00d993"
DEFAULT_CONDS_SHA256 = "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e"
PKUSEG_ARCHIVE_SHA256 = "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe"
SPACE_REVISION = "c612a94210c0af895d6b649850d15b7ccca36594"
LANGUAGES = "ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split()
SAMPLE_RATE = 24000
DECODER_LIMIT_SECONDS = 38.8
DEFAULTS = {
    "seed": 0, "exaggeration": 0.5, "cfg_weight": 0.5,
    "temperature": 0.8, "repetition_penalty": 2.0,
    "min_p": 0.05, "top_p": 1.0, "speed": 1.0,
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _log_failure(stage: str, exc: Exception) -> None:
    """Private diagnostic locations without exception text, source, or locals."""
    frames = [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
              for frame in traceback.extract_tb(exc.__traceback__)]
    print(json.dumps({"event": "chatterbox_failure", "stage": stage,
                      "exception": type(exc).__name__, "frames": frames}),
          file=sys.stderr, flush=True)


@dataclass
class Settings:
    api_key: str
    data_dir: Path
    voice_files: dict[str, Path] = field(default_factory=dict)
    default_voice: str = SPACE_EN_VOICE
    max_chars: int = 250
    queue_limit: int = 16
    generation_timeout: float = 900
    max_audio_seconds: float = 90
    cpu_threads: int = 8
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "Settings":
        reference = os.environ.get("CHATTERBOX_JANE_REFERENCE", "")
        space_reference = os.environ.get("CHATTERBOX_SPACE_EN_REFERENCE", "")
        voice_files = {"jane": Path(reference)} if reference else {}
        if space_reference:
            voice_files[SPACE_EN_VOICE] = Path(space_reference)
        return cls(
            api_key=os.environ.get("CHATTERBOX_API_KEY", ""),
            data_dir=Path(os.environ.get("CHATTERBOX_DATA_DIR", "/data")),
            voice_files=voice_files,
            max_chars=int(os.environ.get("CHATTERBOX_MAX_CHARS", "250")),
            queue_limit=int(os.environ.get("CHATTERBOX_QUEUE_LIMIT", "16")),
            generation_timeout=float(os.environ.get("CHATTERBOX_GENERATION_TIMEOUT", "900")),
            max_audio_seconds=float(os.environ.get("CHATTERBOX_MAX_AUDIO_SECONDS", "90")),
            cpu_threads=int(os.environ.get("CHATTERBOX_CPU_THREADS", "8")),
            device=os.environ.get("CHATTERBOX_DEVICE", "cpu"),
        )


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=1000)
    language_id: str = "en"
    voice: str | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    exaggeration: float = Field(default=0.5, ge=0, le=1)
    cfg_weight: float = Field(default=0.5, ge=0, le=1)
    temperature: float = Field(default=0.8, gt=0, le=2)
    repetition_penalty: float = Field(default=2.0, ge=1, le=3)
    min_p: float = Field(default=0.05, ge=0, le=1)
    top_p: float = Field(default=1.0, gt=0, le=1)
    speed: float = Field(default=1.0, ge=1, le=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("text must contain speech and no NUL bytes")
        # Preserve the exact source. Never slice or silently normalize it.
        return value

    @field_validator("language_id")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in LANGUAGES:
            raise ValueError("unsupported language_id")
        return value


def _cached_snapshot():
    from huggingface_hub import snapshot_download

    # HF_HUB_OFFLINE alone did not suppress list_repo_tree in Hub 1.33.0.
    # Deployment requires verified prefetch; explicitly forbid all Hub I/O.
    return snapshot_download(
        "ResembleAI/chatterbox", revision=MODEL_REVISION, local_files_only=True,
        allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                        "Cangjie5_TC.json"],
    )


def _load_model(settings: Settings):
    """Load only verified local assets; imports and initialization can fail."""
    from prefetch_pkuseg import require_cached_download, verify_cache
    from apply_space_alignment import verify_patch

    verify_patch()

    pkuseg_home = Path(os.environ.get("PKUSEG_HOME", "/cache/pkuseg"))
    os.environ["PKUSEG_HOME"] = str(pkuseg_home)
    # The multilingual tokenizer eagerly initializes PKUSEG for every language.
    # Verify both its pinned archive and extracted model before importing the
    # framework so pkuseg() never downloads missing files during a request.
    pkuseg_cache = verify_cache(pkuseg_home)
    if pkuseg_cache["archive_sha256"] != PKUSEG_ARCHIVE_SHA256:
        raise RuntimeError("Pinned PKUSEG model revision mismatch")

    import torch
    import spacy_pkuseg
    # Even if files disappear after verification, model loading must fail
    # locally instead of falling back to PKUSEG's implicit HTTP downloader.
    spacy_pkuseg.download_model = require_cached_download
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    import chatterbox.models.tokenizers.tokenizer as tokenizer_module

    torch.set_num_threads(settings.cpu_threads)
    snapshot = _cached_snapshot()
    if file_hash(Path(snapshot) / "conds.pt") != DEFAULT_CONDS_SHA256:
        raise RuntimeError("Pinned default voice checksum mismatch")
    # Upstream's tokenizer downloader otherwise resolves Cangjie from mutable
    # main even when from_local receives a pinned checkpoint directory.
    original_download = tokenizer_module.hf_hub_download

    def pinned_tokenizer_download(*args, **kwargs):
        if kwargs.get("repo_id") == "ResembleAI/chatterbox" and kwargs.get("filename") == "Cangjie5_TC.json":
            return str(Path(snapshot) / "Cangjie5_TC.json")
        return original_download(*args, **kwargs)

    tokenizer_module.hf_hub_download = pinned_tokenizer_download
    try:
        model = ChatterboxMultilingualTTS.from_local(snapshot, device=settings.device, t3_model="v2")
    finally:
        tokenizer_module.hf_hub_download = original_download
    if model.sr != SAMPLE_RATE:
        raise RuntimeError("Unexpected model sample rate")
    # HF 5.2 SDPA does not return attention weights. Set the whole backbone
    # before forward so both its causal-mask creation and heads use eager.
    model.t3.tfmr.set_attn_implementation("eager")
    if model.t3.tfmr.config._attn_implementation != "eager":
        raise RuntimeError("Space alignment requires eager attention")
    return model


def _model_child(connection, settings: Settings) -> None:
    """Load lazily and keep model/RNG state confined to the serialized child."""
    # Consume the submitted command first: even a fast initialization failure
    # can then report its cause instead of racing the parent's initial send.
    command = connection.recv()
    if command is None:
        return
    try:
        model = _load_model(settings)
        import random
        import numpy as np
        import soundfile as sf
        import torch

        original_conditioning = copy.deepcopy(model.conds)
    except Exception as exc:
        _log_failure("model_load", exc)
        connection.send({"ok": False, "error": "model_load_failed", "exception": type(exc).__name__})
        return
    while command is not None:
        request, output = command
        try:
            # An earlier reference request must not contaminate the default voice.
            model.conds = copy.deepcopy(original_conditioning)
            random.seed(request["seed"])
            np.random.seed(request["seed"])
            torch.manual_seed(request["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request["seed"])
            kwargs = {name: request[name] for name in (
                "language_id", "exaggeration", "cfg_weight", "temperature",
                "repetition_penalty", "min_p", "top_p",
            )}
            reference = settings.voice_files.get(request["voice"])
            if reference is not None:
                kwargs["audio_prompt_path"] = str(reference)
            with torch.inference_mode():
                samples = model.generate(request["text"], **kwargs).detach().cpu().numpy()
            samples = samples.reshape(-1)
            if not samples.size or not np.isfinite(samples).all():
                raise ValueError("Model produced empty or non-finite audio")
            if samples.size > SAMPLE_RATE * settings.max_audio_seconds:
                raise ValueError("Model output exceeded duration limit")
            sf.write(output, samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            connection.send({"ok": True})
        except Exception as exc:
            # Avoid returning exception strings that may contain local paths/tokens.
            _log_failure("generation", exc)
            connection.send({"ok": False, "error": "generation_failed", "exception": type(exc).__name__})
        command = connection.recv()


class ModelProcess:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.connection = None
        self.stopping = threading.Event()

    def run(self, request: dict, output: Path) -> None:
        if self.stopping.is_set():
            raise RuntimeError("service_stopping")
        if self.process is None or not self.process.is_alive():
            self._terminate()
            context = multiprocessing.get_context("spawn")
            self.connection, child_connection = context.Pipe()
            self.process = context.Process(target=_model_child, args=(child_connection, self.settings), daemon=True)
            self.process.start()
            child_connection.close()
        self.connection.send((request, str(output)))
        deadline = time.monotonic() + self.settings.generation_timeout
        while time.monotonic() < deadline:
            if self.stopping.is_set():
                raise RuntimeError("service_stopping")
            if self.connection.poll(0.2):
                self._receive_response()
                return
            if not self.process.is_alive():
                # A response can arrive after the timed poll returned false
                # but before the child's exit becomes visible. Drain it first.
                if self.connection.poll(0):
                    self._receive_response()
                    return
                self._terminate()
                raise RuntimeError("model_process_exited")
        self._terminate()
        raise TimeoutError("generation_timeout")

    def _receive_response(self):
        try:
            response = self.connection.recv()
        except EOFError:
            self._terminate()
            raise RuntimeError("model_process_exited") from None
        if not response.get("ok"):
            error = response.get("error", "generation_failed")
            if error == "model_load_failed":
                # This child is exiting after its load failure. Reap it
                # before another job can send to the still-alive pipe.
                self._terminate()
            raise RuntimeError(error)

    def _terminate(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=5)
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def close(self):
        self.stopping.set()


class Jobs:
    def __init__(self, settings: Settings, runner=None):
        if not settings.api_key:
            raise RuntimeError("CHATTERBOX_API_KEY must be configured")
        if not 1 <= settings.max_chars <= 1000 or settings.queue_limit < 1:
            raise RuntimeError("Invalid request/queue limits")
        if settings.generation_timeout <= 0 or settings.max_audio_seconds <= 0 or settings.cpu_threads < 1:
            raise RuntimeError("Invalid generation limits")
        for voice, path in settings.voice_files.items():
            if voice not in {"jane", SPACE_EN_VOICE} or not path.is_file():
                raise RuntimeError("Only existing server-owned jane/space-en-f1 references are supported")
            if voice == SPACE_EN_VOICE:
                verify_reference(path)
        if settings.default_voice != "default" and settings.default_voice not in settings.voice_files:
            raise RuntimeError("Default voice reference is not installed; prefetch the official Space reference")
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        # A second HTTP worker/container on the same store must not duplicate jobs.
        import fcntl

        self.owner = (settings.data_dir / ".owner.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("Another service owns this job directory") from None
        self.settings = settings
        self.lock = threading.RLock()
        self.db = sqlite3.connect(settings.data_dir / "jobs.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE, request_hash TEXT, body TEXT)")
        self.db.commit()
        self.manifest = {
            "source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
            "model_variant": "v2", "sample_rate": SAMPLE_RATE,
            "default_voice": settings.default_voice,
            "channels": 1, "sample_width": 2, "encoding": "PCM_16", "defaults": DEFAULTS,
            "max_decoder_seconds": DECODER_LIMIT_SECONDS,
            "device": settings.device, "cpu_threads": settings.cpu_threads,
            "watermark": "upstream_perth", "perth_revision": PERTH_REVISION, "service_contract": 1,
            "pkuseg_archive_sha256": PKUSEG_ARCHIVE_SHA256,
            "space_revision": SPACE_REVISION, "attention_implementation": "eager",
            "alignment_sha256": file_hash(Path(__file__).with_name("space_alignment.py")),
            "alignment_patcher_sha256": file_hash(Path(__file__).with_name("apply_space_alignment.py")),
            "alignment_patch_manifest_sha256": file_hash(Path(__file__).with_name("alignment_patch_manifest.json"))
            if Path(__file__).with_name("alignment_patch_manifest.json").is_file() else None,
            "service_sha256": file_hash(Path(__file__)),
            "pkuseg_loader_sha256": file_hash(Path(__file__).with_name("prefetch_pkuseg.py")),
            "reference_loader_sha256": file_hash(Path(__file__).with_name("prefetch_reference.py")),
            "runtime_requirements_sha256": file_hash(Path("/app/runtime-requirements.txt"))
            if Path("/app/runtime-requirements.txt").is_file() else None,
        }
        self.voices = [{"id": "default", "reference_sha256": DEFAULT_CONDS_SHA256}]
        for voice, path in sorted(settings.voice_files.items()):
            entry = {"id": voice, "reference_sha256": file_hash(path)}
            if voice == SPACE_EN_VOICE:
                entry["reference_source"] = reference_source()
                self.voices.insert(0, entry)
            else:
                self.voices.append(entry)
        for voice in self.voices:
            voice["fingerprint"] = digest({**self.manifest, **voice})
        self.fingerprint = digest({**self.manifest, "voices": self.voices})
        self.runner = runner if runner is not None else ModelProcess(settings)
        self.pending = queue.Queue()
        self.stop = threading.Event()
        with self.lock:
            rows = self.db.execute("SELECT body FROM jobs").fetchall()
            for (body,) in rows:
                job = json.loads(body)
                if job["status"] in {"running", "queued"}:
                    # A restart never silently repeats an uncertain generation.
                    job.update(status="failed", error="service_restarted", finished_at=time.time())
                    self._save(job)
        self.worker = threading.Thread(target=self._work, name="chatterbox-jobs", daemon=True)
        self.worker.start()

    def _save(self, job):
        self.db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(job), job["id"]))
        self.db.commit()

    def get(self, job_id):
        with self.lock:
            row = self.db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found")
        return json.loads(row[0])

    def submit(self, request: Request, key: str):
        body = request.model_dump()
        if body["voice"] is None:
            body["voice"] = self.settings.default_voice
        if len(request.text) > self.settings.max_chars:
            raise HTTPException(422, f"text exceeds {self.settings.max_chars} characters; split without dropping source text")
        voice = next((v for v in self.voices if v["id"] == body["voice"]), None)
        if voice is None:
            raise HTTPException(422, "Voice is unavailable; see /v1/voices")
        request_hash = digest({"request": body, "fingerprint": self.fingerprint})
        with self.lock:
            existing = self.db.execute("SELECT request_hash, body FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
            if existing:
                if not hmac.compare_digest(existing[0], request_hash):
                    raise HTTPException(409, "Idempotency key is already bound to a different request or model fingerprint")
                return json.loads(existing[1])
            active = self.db.execute("SELECT COUNT(*) FROM jobs WHERE json_extract(body, '$.status') IN ('queued', 'running')").fetchone()[0]
            if active >= self.settings.queue_limit:
                raise HTTPException(429, "Generation queue is full", headers={"Retry-After": "5"})
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id, "job_id": job_id, "status": "queued", "request": body,
                "text_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                "request_sha256": digest(body),
                "fingerprint": self.fingerprint, "voice_fingerprint": voice["fingerprint"],
                "source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
                "model_variant": "v2", "created_at": time.time(),
                "audio_url": f"/v1/jobs/{job_id}/audio",
            }
            self.db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?)", (job_id, key, request_hash, json.dumps(job)))
            self.db.commit()
            self.pending.put(job_id)
            return job

    def _work(self):
        while not self.stop.is_set():
            try:
                job_id = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            job = self.get(job_id)
            temporary = self.settings.data_dir / f"{job_id}.partial.wav"
            final = self.settings.data_dir / f"{job_id}.wav"
            try:
                with self.lock:
                    job.update(status="running", started_at=time.time())
                    self._save(job)
                # Hash again before use so reference mutation fails closed.
                reference = self.settings.voice_files.get(job["request"]["voice"])
                expected_voice = next(v for v in self.voices if v["id"] == job["request"]["voice"])
                if job["request"]["voice"] == SPACE_EN_VOICE:
                    try:
                        verify_reference(reference)
                    except (OSError, RuntimeError):
                        raise RuntimeError("voice_reference_changed") from None
                if reference and file_hash(reference) != expected_voice["reference_sha256"]:
                    raise RuntimeError("voice_reference_changed")
                self.runner.run(job["request"], temporary)
                with wave.open(str(temporary), "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
                        raise ValueError("invalid_audio_format")
                    frames = wav.getnframes()
                    if not 0 < frames <= SAMPLE_RATE * self.settings.max_audio_seconds:
                        raise ValueError("invalid_audio_duration")
                    duration = frames / SAMPLE_RATE
                    if duration >= DECODER_LIMIT_SECONDS:
                        raise ValueError("decoder_duration_limit")
                    if len(wav.readframes(frames)) != frames * wav.getsampwidth() * wav.getnchannels():
                        raise ValueError("invalid_audio_format")
                with temporary.open("rb") as output:
                    os.fsync(output.fileno())
                temporary.replace(final)
                job.update(status="completed", finished_at=time.time(), audio_sha256=file_hash(final),
                           duration_seconds=duration, sample_rate=SAMPLE_RATE, channels=1, sample_width=2, encoding="PCM_16")
            except Exception as exc:
                code = str(exc) if str(exc) in {
                    "generation_timeout", "generation_failed", "model_process_exited", "model_load_failed",
                    "service_stopping", "voice_reference_changed", "invalid_audio_format", "invalid_audio_duration",
                    "decoder_duration_limit",
                } else "generation_failed"
                job.update(status="failed", error=code, finished_at=time.time())
                temporary.unlink(missing_ok=True)
            job["generation_seconds"] = max(0.0, job["finished_at"] - job["started_at"])
            with self.lock:
                self._save(job)
        if hasattr(self.runner, "_terminate"):
            self.runner._terminate()

    def close(self):
        self.stop.set()
        self.runner.close()
        self.worker.join(timeout=15)
        if self.worker.is_alive():
            raise RuntimeError("Job worker did not stop")
        self.db.close()
        self.owner.close()


def create_app(settings: Settings | None = None, runner=None) -> FastAPI:
    configured = settings

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = Jobs(configured or Settings.from_env(), runner=runner)
        try:
            yield
        finally:
            app.state.jobs.close()

    app = FastAPI(title="Chatterbox Multilingual V2", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def auth(x_api_key: str | None = Header(default=None)):
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), app.state.jobs.settings.api_key.encode()):
            raise HTTPException(401, "Invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": app.state.jobs.worker.is_alive(),
                "model_load": "lazy", "source_revision": SOURCE_REVISION,
                "model_revision": MODEL_REVISION, "model_variant": "v2"}

    @app.get("/v1/capabilities", dependencies=[Depends(auth)])
    def capabilities():
        jobs = app.state.jobs
        return {**jobs.manifest, "fingerprint": jobs.fingerprint, "voices": jobs.voices,
                "languages": LANGUAGES, "max_chars": jobs.settings.max_chars,
                "max_audio_seconds": jobs.settings.max_audio_seconds, "speed": 1.0,
                "queue_limit": jobs.settings.queue_limit,
                "generation_timeout": jobs.settings.generation_timeout}

    @app.get("/v1/voices", dependencies=[Depends(auth)])
    def voices():
        return {"voices": app.state.jobs.voices}

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(auth)])
    def submit(request: Request, idempotency_key: str = Header(min_length=1, max_length=200)):
        if not idempotency_key.strip():
            raise HTTPException(422, "Idempotency-Key is required")
        return app.state.jobs.submit(request, idempotency_key)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    def status(job_id: str):
        return app.state.jobs.get(job_id)

    @app.get("/v1/jobs/{job_id}/audio", dependencies=[Depends(auth)])
    def audio(job_id: str):
        jobs = app.state.jobs
        job = jobs.get(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "Audio is not complete")
        path = jobs.settings.data_dir / f"{job['id']}.wav"
        if not path.is_file() or file_hash(path) != job["audio_sha256"]:
            raise HTTPException(410, "Audio is missing or its checksum has changed")
        return FileResponse(path, media_type="audio/wav", headers={
            "X-Audio-SHA256": job["audio_sha256"], "X-Model-Fingerprint": job["fingerprint"],
        })

    return app


app = create_app()
