"""Exercise the deployment HTTP contract without importing Torch or weights."""

import hashlib
import importlib.util
import json
import sqlite3
import sys
import threading
import time
import wave
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_PATH = Path(__file__).resolve().parents[2] / "deploy" / "chatterbox" / "service.py"
sys.path.insert(0, str(SERVICE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("chatterbox_deployment_service", SERVICE_PATH)
service = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = service
SPEC.loader.exec_module(service)
AUTH = {"X-API-Key": "test-only-key"}


class FakeRunner:
    def __init__(self, *, failure=None, release=None, sample_rate=24000, frames=2400, truncate_bytes=0):
        self.requests = []
        self.failure = failure
        self.release = release
        self.sample_rate = sample_rate
        self.frames = frames
        self.truncate_bytes = truncate_bytes
        self.started = threading.Event()

    def run(self, request, output):
        self.requests.append(request)
        self.started.set()
        if self.release:
            assert self.release.wait(timeout=3)
        if self.failure:
            raise self.failure
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"\x01\x00" * self.frames)
        if self.truncate_bytes:
            output.write_bytes(output.read_bytes()[:-self.truncate_bytes])

    def close(self):
        if self.release:
            self.release.set()


def settings(tmp_path, **kwargs):
    kwargs.setdefault("default_voice", "default")  # Existing model-default contract tests.
    return service.Settings(api_key="test-only-key", data_dir=tmp_path, **kwargs)


def post(client, text="The whole source is preserved.", key="one", **kwargs):
    return client.post("/v1/jobs", json={"text": text, **kwargs}, headers={**AUTH, "Idempotency-Key": key})


def done(client, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
        if result["status"] in {"completed", "failed"}:
            return result
        time.sleep(0.01)
    raise AssertionError("Job did not terminate")


def test_startup_requires_auth_and_single_store_owner(tmp_path):
    with pytest.raises(RuntimeError, match="API_KEY"):
        service.Jobs(service.Settings(api_key="", data_dir=tmp_path), FakeRunner())
    jobs = service.Jobs(settings(tmp_path), FakeRunner())
    try:
        with pytest.raises(RuntimeError, match="Another service owns"):
            service.Jobs(settings(tmp_path), FakeRunner())
    finally:
        jobs.close()


def test_auth_capabilities_and_validation(tmp_path):
    with TestClient(service.create_app(settings(tmp_path), FakeRunner())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/capabilities").status_code == 401
        assert client.get("/v1/voices", headers={"X-API-Key": "wrong"}).status_code == 401
        caps = client.get("/v1/capabilities", headers=AUTH).json()
        assert (caps["source_revision"], caps["model_revision"], caps["model_variant"]) == (
            service.SOURCE_REVISION, service.MODEL_REVISION, "v2",
        )
        assert caps["sample_rate"] == 24000 and caps["sample_width"] == 2
        assert caps["speed"] == 1 and caps["defaults"]["repetition_penalty"] == 2
        assert caps["max_decoder_seconds"] == 38.8
        assert caps["pkuseg_archive_sha256"] == service.PKUSEG_ARCHIVE_SHA256
        assert caps["space_revision"] == service.SPACE_REVISION
        assert caps["attention_implementation"] == "eager"
        assert len(caps["alignment_sha256"]) == len(caps["alignment_patcher_sha256"]) == 64
        assert client.post("/v1/jobs", json={"text": "hello"}, headers=AUTH).status_code == 422
        for changes in ({"speed": 1.2}, {"language_id": "xx"}, {"voice": "../../secret"},
                        {"voice": "jane"}, {"seed": -1}, {"extra": "ignored?"}):
            assert post(client, **changes).status_code == 422
        assert post(client, text="\n ").status_code == 422
        assert post(client, text="A" * 251).status_code == 422


def test_exact_source_pcm_and_durable_idempotency(tmp_path):
    runner = FakeRunner()
    source = "  AI uses twenty-four thousand samples.\nNo lost words.  "
    with TestClient(service.create_app(settings(tmp_path), runner)) as client:
        response = post(client, text=source)
        assert response.status_code == 202
        job = done(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["request"]["text"] == source == runner.requests[0]["text"]
        assert job["text_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        caps = client.get("/v1/capabilities", headers=AUTH).json()
        assert job["fingerprint"] == caps["fingerprint"]
        assert job["voice_fingerprint"] == caps["voices"][0]["fingerprint"]
        audio = client.get(job["audio_url"], headers=AUTH)
        assert audio.status_code == 200
        assert hashlib.sha256(audio.content).hexdigest() == job["audio_sha256"]
        assert post(client, text=source).json()["id"] == job["id"]
        assert post(client, text=source + " changed").status_code == 409
        assert len(runner.requests) == 1
        (tmp_path / f"{job['id']}.wav").write_bytes(b"corrupted")
        assert client.get(job["audio_url"], headers=AUTH).status_code == 410
    restarted = FakeRunner()
    with TestClient(service.create_app(settings(tmp_path), restarted)) as client:
        assert post(client, text=source).json()["id"] == job["id"]
        assert restarted.requests == []


@pytest.mark.parametrize("failure,rate,error", [
    (TimeoutError("generation_timeout"), 24000, "generation_timeout"),
    (RuntimeError("model_load_failed"), 24000, "model_load_failed"),
    (RuntimeError("secret detail /home/private"), 24000, "generation_failed"),
    (None, 22050, "invalid_audio_format"),
])
def test_failures_are_durable_and_do_not_return_partial_audio(tmp_path, failure, rate, error):
    runner = FakeRunner(failure=failure, sample_rate=rate)
    with TestClient(service.create_app(settings(tmp_path), runner)) as client:
        job = done(client, post(client).json()["id"])
        assert job["status"] == "failed" and job["error"] == error
        assert client.get(job["audio_url"], headers=AUTH).status_code == 409
        assert post(client).json()["id"] == job["id"]
        assert len(runner.requests) == 1
        assert not list(tmp_path.glob("*.partial.wav"))


@pytest.mark.parametrize("frames,accepted", [(931199, True), (931200, False), (931201, False)])
def test_decoder_duration_ceiling_is_rejected_before_promotion(tmp_path, frames, accepted):
    with TestClient(service.create_app(settings(tmp_path), FakeRunner(frames=frames))) as client:
        job = done(client, post(client).json()["id"])
        if accepted:
            assert job["status"] == "completed"
            assert job["duration_seconds"] < 38.8
            assert client.get(job["audio_url"], headers=AUTH).status_code == 200
        else:
            assert job["status"] == "failed" and job["error"] == "decoder_duration_limit"
            assert client.get(job["audio_url"], headers=AUTH).status_code == 409
            assert not list(tmp_path.glob("*.wav"))


def test_wav_declared_frames_require_matching_physical_samples(tmp_path):
    with TestClient(service.create_app(settings(tmp_path), FakeRunner(truncate_bytes=100))) as client:
        job = done(client, post(client).json()["id"])
        assert job["status"] == "failed" and job["error"] == "invalid_audio_format"
        assert client.get(job["audio_url"], headers=AUTH).status_code == 409
        assert not list(tmp_path.glob("*.wav"))


def test_queue_bounds_count_running_and_admit_idempotent_replay(tmp_path):
    release = threading.Event()
    runner = FakeRunner(release=release)
    with TestClient(service.create_app(settings(tmp_path, queue_limit=1), runner)) as client:
        first = post(client).json()
        assert runner.started.wait(1)
        assert post(client).json()["id"] == first["id"]
        assert post(client, key="two").status_code == 429
        release.set()
        assert done(client, first["id"])["status"] == "completed"


def test_reference_hash_changes_fail_closed_and_change_fingerprint(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-a")
    runner = FakeRunner()
    with TestClient(service.create_app(settings(tmp_path, voice_files={"jane": reference}), runner)) as client:
        before = client.get("/v1/capabilities", headers=AUTH).json()
        jane = next(v for v in before["voices"] if v["id"] == "jane")
        assert jane["reference_sha256"] == hashlib.sha256(b"reference-a").hexdigest()
        reference.write_bytes(b"reference-b")
        job = done(client, post(client, voice="jane").json()["id"])
        assert job["error"] == "voice_reference_changed"
        assert runner.requests == []
    with TestClient(service.create_app(settings(tmp_path, voice_files={"jane": reference}), FakeRunner())) as client:
        after = client.get("/v1/capabilities", headers=AUTH).json()
        assert after["fingerprint"] != before["fingerprint"]
        assert post(client, voice="jane").status_code == 409


@pytest.mark.parametrize("interrupted_status", ["queued", "running"])
def test_restart_marks_interrupted_jobs_failed_without_replaying(tmp_path, interrupted_status):
    with TestClient(service.create_app(settings(tmp_path), FakeRunner())) as client:
        job = done(client, post(client).json()["id"])
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        job["status"] = interrupted_status
        db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(job), job["id"]))
    runner = FakeRunner()
    with TestClient(service.create_app(settings(tmp_path), runner)) as client:
        recovered = client.get(f"/v1/jobs/{job['id']}", headers=AUTH).json()
        assert recovered["status"] == "failed" and recovered["error"] == "service_restarted"
        assert post(client).json()["id"] == job["id"]
        assert runner.requests == []


def test_generation_deadline_terminates_child(tmp_path):
    class Process:
        alive = True
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.alive = False
        def join(self, timeout):
            pass

    class Connection:
        def send(self, data):
            pass
        def poll(self, timeout):
            return False
        def close(self):
            pass

    runner = service.ModelProcess(settings(tmp_path, generation_timeout=0.005))
    process = Process()
    runner.process, runner.connection = process, Connection()
    with pytest.raises(TimeoutError, match="generation_timeout"):
        runner.run({}, tmp_path / "unused.wav")
    assert not process.alive and runner.process is None


def test_model_snapshot_explicitly_forbids_hub_network_calls(monkeypatch):
    from types import SimpleNamespace

    calls = []
    def snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["local_files_only"] is True
        return "/cache/pinned-snapshot"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    assert service._cached_snapshot() == "/cache/pinned-snapshot"
    assert calls[0][0] == ("ResembleAI/chatterbox",)
    assert calls[0][1]["revision"] == service.MODEL_REVISION
    assert len(calls[0][1]["allow_patterns"]) == 6


def test_model_initialization_failure_is_explicit_and_does_not_leak_details(tmp_path, monkeypatch, capsys):
    class Connection:
        received = 0
        sent = []
        def recv(self):
            self.received += 1
            return ({"text": "test"}, str(tmp_path / "audio.wav"))
        def send(self, response):
            self.sent.append(response)

    loads = []
    def fail_load(configured):
        loads.append(configured)
        raise OSError("private path and secret token must not reach HTTP")

    monkeypatch.setattr(service, "_load_model", fail_load)
    connection = Connection()
    service._model_child(connection, settings(tmp_path))
    assert len(loads) == 1 and connection.received == 1
    assert connection.sent == [{"ok": False, "error": "model_load_failed", "exception": "OSError"}]
    assert not (tmp_path / "audio.wav").exists()
    stderr = capsys.readouterr().err
    diagnostic = json.loads(stderr)
    assert diagnostic["stage"] == "model_load" and diagnostic["exception"] == "OSError"
    assert diagnostic["frames"][-1]["function"] == "fail_load"
    assert "secret token" not in stderr and "private path" not in stderr
    assert all(set(frame) == {"file", "line", "function"} for frame in diagnostic["frames"])


def test_private_generation_diagnostic_omits_exception_text_and_locals(capsys):
    secret_local = "private-local-value"
    try:
        raise ValueError("private-exception-message " + secret_local)
    except ValueError as exc:
        service._log_failure("generation", exc)
    stderr = capsys.readouterr().err
    diagnostic = json.loads(stderr)
    assert diagnostic["stage"] == "generation" and diagnostic["exception"] == "ValueError"
    assert diagnostic["frames"]
    assert "private-exception-message" not in stderr and secret_local not in stderr


@pytest.mark.parametrize("error,reaped", [("model_load_failed", True), ("generation_failed", False)])
def test_load_failure_reaps_exiting_worker_but_generation_failure_keeps_model(tmp_path, error, reaped):
    class Process:
        alive = True
        joined = False
        def is_alive(self):
            # Simulate the race: child has sent a failure but has not exited.
            return self.alive
        def terminate(self):
            self.alive = False
        def join(self, timeout):
            self.joined = True

    class Connection:
        closed = False
        def send(self, request):
            assert not self.closed
        def poll(self, timeout):
            return True
        def recv(self):
            return {"ok": False, "error": error}
        def close(self):
            self.closed = True

    runner = service.ModelProcess(settings(tmp_path))
    process, connection = Process(), Connection()
    runner.process, runner.connection = process, connection
    with pytest.raises(RuntimeError, match=error):
        runner.run({}, tmp_path / "output.wav")
    if reaped:
        assert process.joined and not process.alive and connection.closed
        assert runner.process is None and runner.connection is None
    else:
        assert not process.joined and process.alive and not connection.closed
        assert runner.process is process and runner.connection is connection


def test_child_exit_after_timed_poll_drains_pending_load_failure(tmp_path):
    class Process:
        checks = 0
        def is_alive(self):
            self.checks += 1
            return self.checks == 1

    class Connection:
        closed = False
        polls = []
        received = False
        def send(self, request):
            pass
        def poll(self, timeout):
            self.polls.append(timeout)
            return timeout == 0
        def recv(self):
            self.received = True
            return {"ok": False, "error": "model_load_failed", "exception": "OSError"}
        def close(self):
            self.closed = True

    runner = service.ModelProcess(settings(tmp_path))
    connection = Connection()
    runner.process, runner.connection = Process(), connection
    with pytest.raises(RuntimeError, match="model_load_failed"):
        runner.run({}, tmp_path / "unused.wav")
    assert connection.polls == [0.2, 0]
    assert connection.received and connection.closed
    assert runner.process is None and runner.connection is None


def _prefetch_module():
    spec = importlib.util.spec_from_file_location("chatterbox_prefetch", SERVICE_PATH.with_name("prefetch_models.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefetch_resumes_verifies_and_installs_standard_hf_cache(tmp_path, monkeypatch):
    prefetch = _prefetch_module()
    content = b"complete pinned weights"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (tmp_path / "blobs").mkdir()
    (tmp_path / "snapshots" / prefetch.REVISION).mkdir(parents=True)
    partial = tmp_path / "blobs" / (expected["sha256"] + ".incomplete")
    partial.write_bytes(content[:5])
    calls = []

    def fake_curl(args):
        calls.append(args)
        assert args[args.index("--continue-at") + 1] == "-"
        assert prefetch.REVISION in args[-1]
        assert partial.read_bytes() == content[:5]
        partial.write_bytes(content)

    monkeypatch.setattr(prefetch.subprocess, "check_call", fake_curl)
    prefetch.fetch_file(tmp_path, expected, False)
    assert not partial.exists()
    assert (tmp_path / "snapshots" / prefetch.REVISION / "example.pt").read_bytes() == content
    prefetch.fetch_file(tmp_path, expected, True)
    assert len(calls) == 1


@pytest.mark.parametrize("suffix", ["", ".incomplete"])
def test_prefetch_preserves_invalid_existing_files(tmp_path, monkeypatch, suffix):
    prefetch = _prefetch_module()
    expected = {"name": "example.pt", "size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    (tmp_path / "blobs").mkdir()
    (tmp_path / "snapshots" / prefetch.REVISION).mkdir(parents=True)
    bad = tmp_path / "blobs" / (expected["sha256"] + suffix)
    bad.write_bytes(b"evil")
    monkeypatch.setattr(prefetch.subprocess, "check_call", lambda *_: pytest.fail("must preserve invalid files"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prefetch.fetch_file(tmp_path, expected, False)
    assert bad.read_bytes() == b"evil"


def _fake_range_response(args, content, *, status=206, wrong_range=False, wrong_body=False):
    from types import SimpleNamespace

    start, end = [int(value) for value in args[args.index("--range") + 1].split("-")]
    body = content[start:end + 1]
    if wrong_body:
        body = b"x" * len(body)
    Path(args[args.index("--output") + 1]).write_bytes(body)
    headers = ("HTTP/1.1 302 Found\r\nLocation: https://signed.example/secret-token\r\n\r\n"
               "HTTP/2 {}\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\n\r\n").format(
                   status, start + int(wrong_range), end, len(content), len(body))
    return SimpleNamespace(returncode=0, stdout=headers.encode(), stderr=b"")


def test_parallel_ranges_reuse_prefix_and_completed_fragments(tmp_path, monkeypatch, capsys):
    prefetch = _prefetch_module()
    content = b"abcdefghijklmnopqrstuvwxyz012345"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    partial = tmp_path / "weights.incomplete"
    partial.write_bytes(content[:5])
    directory = tmp_path / "weights.incomplete.ranges"
    directory.mkdir()
    previous = directory / "000000000008-000000000015.part"
    previous.write_bytes(content[8:16])
    calls = []

    def fake_curl(args, **kwargs):
        calls.append(args[args.index("--range") + 1])
        assert kwargs == {"stdout": prefetch.subprocess.PIPE, "stderr": prefetch.subprocess.PIPE}
        return _fake_range_response(args, content)

    monkeypatch.setattr(prefetch.subprocess, "run", fake_curl)
    prefetch.download_ranges(partial, expected, connections=3, range_bytes=8)
    assert partial.read_bytes() == content
    assert sorted(calls) == ["16-23", "24-31", "5-7"]
    assert previous.read_bytes() == content[8:16]
    assert "secret-token" not in capsys.readouterr().out


@pytest.mark.parametrize("status,wrong_range,wrong_body,error", [
    (200, False, False, "exact range"),
    (206, True, False, "exact range"),
    (206, False, True, "full size/SHA256"),
])
def test_parallel_ranges_reject_ignored_changed_or_corrupt_ranges(
    tmp_path, monkeypatch, status, wrong_range, wrong_body, error,
):
    prefetch = _prefetch_module()
    content = b"abcdefghijklmnop"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    partial = tmp_path / "weights.incomplete"
    partial.write_bytes(content[:3])
    monkeypatch.setattr(prefetch.subprocess, "run", lambda args, **_: _fake_range_response(
        args, content, status=status, wrong_range=wrong_range, wrong_body=wrong_body,
    ))
    with pytest.raises(RuntimeError, match=error):
        prefetch.download_ranges(partial, expected, connections=2, range_bytes=8)
    assert partial.read_bytes() == content[:3]


def test_range_interruption_keeps_validated_bytes_and_requests_only_missing_tail(tmp_path, monkeypatch):
    prefetch = _prefetch_module()
    content = b"0123456789abcdef"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    calls = []

    def interrupted_then_complete(args, **kwargs):
        assert "--http1.1" in args
        calls.append(args[args.index("--range") + 1])
        response = _fake_range_response(args, content)
        if len(calls) == 1:
            Path(args[args.index("--output") + 1]).write_bytes(content[4:7])
            response.returncode = 18
        return response

    monkeypatch.setattr(prefetch.subprocess, "run", interrupted_then_complete)
    with pytest.raises(RuntimeError, match="preserved 3 candidate bytes"):
        prefetch._fetch_range(expected, tmp_path, 4, 11)
    staged = tmp_path / "000000000004-000000000011.part.downloading"
    assert staged.read_bytes() == content[4:7]
    fragment = prefetch._fetch_range(expected, tmp_path, 4, 11)
    assert calls == ["4-11", "7-11"]
    assert fragment.read_bytes() == content[4:12]
    record = json.loads(staged.with_name(staged.name + ".json").read_text())
    assert record["requires_full_model_sha256"] is True
    assert record["unverified_candidate_prefix_bytes"] == 0


def test_old_range_prefix_is_a_candidate_and_is_not_downloaded_again(tmp_path, monkeypatch):
    prefetch = _prefetch_module()
    content = b"0123456789abcdef"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    staged = tmp_path / "000000000004-000000000011.part.downloading"
    staged.write_bytes(content[4:8])
    calls = []

    def fake_curl(args, **kwargs):
        calls.append(args[args.index("--range") + 1])
        return _fake_range_response(args, content)

    monkeypatch.setattr(prefetch.subprocess, "run", fake_curl)
    result = prefetch._fetch_range(expected, tmp_path, 4, 11)
    assert calls == ["8-11"]
    assert result.read_bytes() == content[4:12]
    record = json.loads(staged.with_name(staged.name + ".json").read_text())
    assert record["unverified_candidate_prefix_bytes"] == 4
    assert record["requires_full_model_sha256"] is True


@pytest.mark.parametrize("status,wrong_range", [(200, False), (206, True)])
def test_bad_tail_response_never_joins_the_saved_range_prefix(tmp_path, monkeypatch, status, wrong_range):
    prefetch = _prefetch_module()
    content = b"0123456789abcdef"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    staged = tmp_path / "000000000004-000000000011.part.downloading"
    staged.write_bytes(content[4:8])
    monkeypatch.setattr(prefetch.subprocess, "run", lambda args, **_: _fake_range_response(
        args, content, status=status, wrong_range=wrong_range,
    ))
    with pytest.raises(RuntimeError, match="exact range"):
        prefetch._fetch_range(expected, tmp_path, 4, 11)
    assert staged.read_bytes() == content[4:8]
    assert len(list(tmp_path.glob("*.tail-*"))) == 1
    assert not list(tmp_path.glob("*.part"))


def test_oversized_range_prefix_is_preserved_without_http(tmp_path, monkeypatch):
    prefetch = _prefetch_module()
    expected = {"name": "example.pt", "size": 16, "sha256": "unused"}
    staged = tmp_path / "000000000004-000000000011.part.downloading"
    staged.write_bytes(b"too many bytes")
    monkeypatch.setattr(prefetch.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must reject before HTTP"))
    with pytest.raises(RuntimeError, match="longer than"):
        prefetch._fetch_range(expected, tmp_path, 4, 11)
    assert staged.read_bytes() == b"too many bytes"


def test_corrupt_legacy_candidate_cannot_pass_whole_model_acceptance(tmp_path, monkeypatch):
    prefetch = _prefetch_module()
    content = b"0123456789abcdef"
    expected = {"name": "example.pt", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    partial = tmp_path / "model.incomplete"
    ranges = tmp_path / "model.incomplete.ranges"
    ranges.mkdir()
    (ranges / "000000000000-000000000007.part.downloading").write_bytes(b"bad")
    monkeypatch.setattr(prefetch.subprocess, "run", lambda args, **_: _fake_range_response(args, content))
    with pytest.raises(RuntimeError, match="full size/SHA256"):
        prefetch.download_ranges(partial, expected, connections=2, range_bytes=8)
    assert not partial.exists()
    assert (ranges / "000000000000-000000000007.part").read_bytes() == b"bad" + content[3:8]


def _pkuseg_module():
    spec = importlib.util.spec_from_file_location("chatterbox_pkuseg", SERVICE_PATH.with_name("prefetch_pkuseg.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pkuseg_fixture(tmp_path, monkeypatch, member=None):
    pkuseg = _pkuseg_module()
    archive = tmp_path / pkuseg.ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as zipped:
        if member is not None:
            zipped.writestr(member, b"unsafe")
        else:
            zipped.writestr("features.msgpack", b"pinned feature bytes")
            zipped.writestr("weights.npz", b"pinned weight bytes")
    monkeypatch.setattr(pkuseg, "ARCHIVE_SIZE", archive.stat().st_size)
    monkeypatch.setattr(pkuseg, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(pkuseg.subprocess, "check_call", lambda *_: pytest.fail("valid cached assets must not trigger HTTP"))
    return pkuseg


def test_pkuseg_prefetch_reuses_verified_archive_and_model_without_http(tmp_path, monkeypatch):
    pkuseg = _pkuseg_fixture(tmp_path, monkeypatch)
    report = pkuseg.prepare_cache(tmp_path)
    weights = tmp_path / pkuseg.MODEL_NAME / "weights.npz"
    assert weights.read_bytes() == b"pinned weight bytes"
    before = weights.stat().st_mtime_ns
    assert pkuseg.prepare_cache(tmp_path) == report == pkuseg.verify_cache(tmp_path)
    assert weights.stat().st_mtime_ns == before
    pkuseg.require_cached_download(pkuseg.ARCHIVE_URL, str(tmp_path), pkuseg.ARCHIVE_SHA256)
    with pytest.raises(RuntimeError, match="Unpinned"):
        pkuseg.require_cached_download("https://unexpected.example/model.zip", str(tmp_path), pkuseg.ARCHIVE_SHA256)


def test_pkuseg_corrupt_extraction_is_preserved_and_load_fails_closed(tmp_path, monkeypatch):
    pkuseg = _pkuseg_fixture(tmp_path, monkeypatch)
    pkuseg.prepare_cache(tmp_path)
    weights = tmp_path / pkuseg.MODEL_NAME / "weights.npz"
    weights.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="missing or changed"):
        pkuseg.prepare_cache(tmp_path)
    with pytest.raises(RuntimeError, match="missing or changed"):
        pkuseg.require_cached_download(pkuseg.ARCHIVE_URL, str(tmp_path), pkuseg.ARCHIVE_SHA256)
    assert weights.read_bytes() == b"tampered"


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:\\escape", "."])
def test_pkuseg_zip_rejects_path_traversal(tmp_path, monkeypatch, name):
    pkuseg = _pkuseg_fixture(tmp_path, monkeypatch, member=name)
    with pytest.raises(RuntimeError, match="Unsafe"):
        pkuseg.prepare_cache(tmp_path)
    assert not (tmp_path / pkuseg.MODEL_NAME).exists()
    assert not list(tmp_path.glob(".pkuseg-extract-*"))


def test_pkuseg_zip_rejects_symlinks(tmp_path, monkeypatch):
    entry = zipfile.ZipInfo("link")
    entry.create_system = 3
    entry.external_attr = 0o120777 << 16
    pkuseg = _pkuseg_fixture(tmp_path, monkeypatch, member=entry)
    with pytest.raises(RuntimeError, match="Unsafe"):
        pkuseg.prepare_cache(tmp_path)
    assert not (tmp_path / pkuseg.MODEL_NAME).exists()


def _alignment_module():
    pytest.importorskip("torch")
    path = SERVICE_PATH.with_name("space_alignment.py")
    spec = importlib.util.spec_from_file_location("deployment_space_alignment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _alignment_fixture(module, *, text_tokens=8, implementation="eager"):
    from types import SimpleNamespace
    import torch

    layers = [SimpleNamespace(self_attn=torch.nn.Identity()) for _ in range(14)]
    model = SimpleNamespace(layers=layers, config=SimpleNamespace(_attn_implementation=implementation))
    analyzer = module.AlignmentStreamAnalyzer(model, None, (0, text_tokens), eos_idx=9)
    return analyzer, model


def _alignment_frame(analyzer, position, token, *, initial=False):
    import torch

    size = analyzer.text_tokens_slice[1]
    rows = size + 1 if initial else 1
    attention = torch.zeros(rows, size + 1)
    attention[-1, position] = 1
    analyzer.last_aligned_attns = [attention.clone() for _ in range(3)]
    return analyzer.step(torch.zeros(1, 12), next_token=token)


def test_space_alignment_suppresses_early_eos_and_requires_fresh_weights():
    module = _alignment_module()
    analyzer, _ = _alignment_fixture(module)
    try:
        assert _alignment_frame(analyzer, 0, 1, initial=True)[0, 9] == -(2**15)
        with pytest.raises(RuntimeError, match="hooks did not run"):
            analyzer.step(module.torch.zeros(1, 12), next_token=2)
    finally:
        analyzer.close()


def test_space_alignment_preserves_two_repeated_tokens_rule_before_completion():
    module = _alignment_module()
    analyzer, _ = _alignment_fixture(module)
    try:
        _alignment_frame(analyzer, 0, 1, initial=True)
        _alignment_frame(analyzer, 0, 2)
        logits = _alignment_frame(analyzer, 0, 2)
        assert not analyzer.complete
        assert logits[0, 9] == 2**15 and logits[0, 8] == -(2**15)
    finally:
        analyzer.close()


@pytest.mark.parametrize("mode", ["long_tail", "return_to_prefix"])
def test_space_alignment_forces_eos_on_tail_or_return_to_previous_text(mode):
    module = _alignment_module()
    analyzer, _ = _alignment_fixture(module)
    try:
        for pos in range(6):
            _alignment_frame(analyzer, pos, pos, initial=pos == 0)
        assert analyzer.complete
        for i in range(6):
            logits = _alignment_frame(analyzer, 5 if mode == "long_tail" else 0, 10 + i)
        assert logits[0, 9] == 2**15 and logits[0, 8] == -(2**15)
    finally:
        analyzer.close()


def test_space_alignment_short_text_and_exception_remove_all_hooks():
    module = _alignment_module()
    from types import SimpleNamespace

    analyzer, model = _alignment_fixture(module, text_tokens=4)
    analyzer.close()
    owner = SimpleNamespace(_space_alignment=None)

    @module.alignment_guard
    def run(target, fail):
        target._space_alignment = module.AlignmentStreamAnalyzer(model, None, (0, 4), eos_idx=9)
        for i in range(3):
            _alignment_frame(target._space_alignment, min(i, 1), i, initial=i == 0)
        if fail:
            raise ValueError("mock inference failure")

    for fail in (False, True, False):
        if fail:
            with pytest.raises(ValueError):
                run(owner, fail)
        else:
            run(owner, fail)
        assert owner._space_alignment is None
        assert all(not layer.self_attn._forward_hooks for layer in model.layers)


def test_space_alignment_rejects_sdpa_and_none_attention():
    module = _alignment_module()
    with pytest.raises(RuntimeError, match="eager"):
        _alignment_fixture(module, implementation="sdpa")
    analyzer, model = _alignment_fixture(module)
    try:
        layer = model.layers[9].self_attn
        hook = next(iter(layer._forward_hooks.values()))
        with pytest.raises(RuntimeError, match="unavailable"):
            hook(layer, (), (module.torch.zeros(1), None))
    finally:
        analyzer.close()


def test_space_alignment_hook_observes_only_conditional_cfg_batch():
    module = _alignment_module()
    analyzer, model = _alignment_fixture(module)
    try:
        layer = model.layers[9].self_attn
        hook = next(iter(layer._forward_hooks.values()))
        weights = module.torch.zeros(2, 16, 1, 9)
        weights[0, 2, 0, 3] = 1
        weights[1, 2, 0, 7] = 1
        output = (module.torch.zeros(2, 1, 32), weights)
        assert hook(layer, (), output) is None
        assert analyzer.last_aligned_attns[2][0, 3] == 1
        assert analyzer.last_aligned_attns[2][0, 7] == 0
    finally:
        analyzer.close()


def test_space_patcher_rejects_unexpected_source():
    path = SERVICE_PATH.with_name("apply_space_alignment.py")
    spec = importlib.util.spec_from_file_location("deployment_space_patcher", path)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    with pytest.raises(RuntimeError, match="Unexpected upstream"):
        patcher.patched_source("unreviewed source")


def _official_reference_fixture(tmp_path, monkeypatch):
    import prefetch_reference

    content = b"test-only original FLAC bytes"
    monkeypatch.setattr(prefetch_reference, "SIZE_BYTES", len(content))
    monkeypatch.setattr(prefetch_reference, "SHA256", hashlib.sha256(content).hexdigest())
    path = tmp_path / "en_f1.flac"
    path.write_bytes(content)
    return prefetch_reference, path, content


def test_production_env_requires_official_reference_and_keeps_jane(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTERBOX_API_KEY", "test-only-key")
    monkeypatch.setenv("CHATTERBOX_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("CHATTERBOX_SPACE_EN_REFERENCE", raising=False)
    monkeypatch.delenv("CHATTERBOX_JANE_REFERENCE", raising=False)
    configured = service.Settings.from_env()
    assert configured.default_voice == "space-en-f1"
    with pytest.raises(RuntimeError, match="Default voice reference"):
        service.Jobs(configured, FakeRunner())
    _, official, _ = _official_reference_fixture(tmp_path, monkeypatch)
    jane = tmp_path / "jane.wav"
    jane.write_bytes(b"retained Pocket reference")
    monkeypatch.setenv("CHATTERBOX_SPACE_EN_REFERENCE", str(official))
    monkeypatch.setenv("CHATTERBOX_JANE_REFERENCE", str(jane))
    configured = service.Settings.from_env()
    assert configured.voice_files == {"jane": jane, "space-en-f1": official}
    with TestClient(service.create_app(configured, FakeRunner())) as client:
        caps = client.get("/v1/capabilities", headers=AUTH).json()
        assert caps["default_voice"] == "space-en-f1"
        assert [voice["id"] for voice in caps["voices"]] == ["space-en-f1", "default", "jane"]


def test_official_default_is_canonical_and_explicit_legacy_voices_keep_identity(tmp_path, monkeypatch):
    reference, official, _ = _official_reference_fixture(tmp_path, monkeypatch)
    jane = tmp_path / "jane.wav"
    jane.write_bytes(b"retained Pocket reference")
    runner = FakeRunner()
    configured = settings(tmp_path, default_voice="space-en-f1", voice_files={"space-en-f1": official, "jane": jane})
    with TestClient(service.create_app(configured, runner)) as client:
        caps = client.get("/v1/capabilities", headers=AUTH).json()
        voices = {v["id"]: v for v in caps["voices"]}
        source = voices["space-en-f1"]["reference_source"]
        assert source["space_revision"] == service.SPACE_REVISION
        assert source["generation"] == "1755807478904629"
        assert source["url"].endswith("en_f1.flac?generation=1755807478904629")
        assert voices["space-en-f1"]["reference_sha256"] == reference.SHA256
        assert voices["default"]["reference_sha256"] == service.DEFAULT_CONDS_SHA256
        assert voices["jane"]["reference_sha256"] == hashlib.sha256(jane.read_bytes()).hexdigest()
        job = done(client, post(client).json()["id"])
        assert job["request"]["voice"] == "space-en-f1"
        assert job["voice_fingerprint"] == voices["space-en-f1"]["fingerprint"]
        assert post(client, voice="space-en-f1").json()["id"] == job["id"]
        assert post(client, voice="default").status_code == 409
        for voice in ("jane", "default"):
            result = done(client, post(client, key=voice, voice=voice).json()["id"])
            assert result["request"]["voice"] == voice
            assert result["voice_fingerprint"] == voices[voice]["fingerprint"]
        assert [r["voice"] for r in runner.requests] == ["space-en-f1", "jane", "default"]


@pytest.mark.parametrize("changed", [b"wrong-size", b"x" * len(b"test-only original FLAC bytes")])
def test_official_reference_corruption_fails_startup_and_each_job(tmp_path, monkeypatch, changed):
    _, official, content = _official_reference_fixture(tmp_path, monkeypatch)
    configured = settings(tmp_path, default_voice="space-en-f1", voice_files={"space-en-f1": official})
    official.write_bytes(changed)
    with pytest.raises(RuntimeError, match="Official Space English reference"):
        service.Jobs(configured, FakeRunner())
    official.write_bytes(content)
    runner = FakeRunner()
    with TestClient(service.create_app(configured, runner)) as client:
        official.write_bytes(changed)
        job = done(client, post(client).json()["id"])
        assert job["status"] == "failed" and job["error"] == "voice_reference_changed"
        assert runner.requests == []
        assert client.get(job["audio_url"], headers=AUTH).status_code == 409
    assert official.read_bytes() == changed


def test_new_official_default_preserves_historical_job_and_rejects_key_rebinding(tmp_path, monkeypatch):
    with TestClient(service.create_app(settings(tmp_path), FakeRunner())) as client:
        historical = done(client, post(client, voice="default").json()["id"])
        audio = client.get(historical["audio_url"], headers=AUTH).content
    _, official, _ = _official_reference_fixture(tmp_path, monkeypatch)
    configured = settings(tmp_path, default_voice="space-en-f1", voice_files={"space-en-f1": official})
    with TestClient(service.create_app(configured, FakeRunner())) as client:
        assert client.get(f"/v1/jobs/{historical['id']}", headers=AUTH).json() == historical
        assert client.get(historical["audio_url"], headers=AUTH).content == audio
        assert post(client).status_code == 409
        current = done(client, post(client, key="new-official-job").json()["id"])
        assert current["request"]["voice"] == "space-en-f1"
        assert current["fingerprint"] != historical["fingerprint"]


def test_reference_prefetch_checks_generation_and_preserves_existing_files(tmp_path, monkeypatch):
    import io

    reference, installed, content = _official_reference_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reference.urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("valid cache must not fetch"))
    assert reference.prepare_reference(installed) == installed
    installed.write_bytes(b"other retained bytes")
    with pytest.raises(RuntimeError):
        reference.prepare_reference(installed)
    assert installed.read_bytes() == b"other retained bytes"

    calls = []
    def fetch(url, timeout):
        calls.append(url)
        stream = io.BytesIO(content)
        stream.headers = {"x-goog-generation": reference.GENERATION}
        return stream
    monkeypatch.setattr(reference.urllib.request, "urlopen", fetch)
    destination = tmp_path / "new.flac"
    assert reference.prepare_reference(destination) == destination
    assert destination.read_bytes() == content
    assert calls == [reference.SOURCE_URL]
    assert reference.verify_reference(destination) == reference.SHA256

    def wrong_generation(*_a, **_k):
        stream = io.BytesIO(content)
        stream.headers = {"x-goog-generation": "different"}
        return stream
    monkeypatch.setattr(reference.urllib.request, "urlopen", wrong_generation)
    with pytest.raises(RuntimeError, match="generation"):
        reference.prepare_reference(tmp_path / "unaccepted.flac")
    assert not (tmp_path / "unaccepted.flac").exists()


def test_model_child_resets_conditioning_between_official_jane_and_bundled_voice(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np

    calls = []
    class Model:
        conds = {"identity": "bundled"}
        def generate(self, text, **kwargs):
            calls.append((dict(self.conds), kwargs.get("audio_prompt_path")))
            self.conds["identity"] = "mutated by preceding generation"
            return SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.ones(24))))
    model = Model()
    monkeypatch.setattr(service, "_load_model", lambda _: model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda _: None, cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=lambda *_a, **_k: None))
    commands = iter([
        (service.Request(text="Complete sentence.", voice=voice).model_dump(), str(tmp_path / f"{voice}.wav"))
        for voice in ("space-en-f1", "jane", "default", "space-en-f1")
    ] + [None])
    sent = []
    connection = SimpleNamespace(recv=lambda: next(commands), send=sent.append)
    service._model_child(connection, settings(tmp_path, voice_files={
        "space-en-f1": tmp_path / "en_f1.flac", "jane": tmp_path / "jane.wav",
    }))
    assert sent == [{"ok": True}] * 4
    assert [conds for conds, _ in calls] == [{"identity": "bundled"}] * 4
    assert [path for _, path in calls] == [str(tmp_path / "en_f1.flac"), str(tmp_path / "jane.wav"), None, str(tmp_path / "en_f1.flac")]
