#!/usr/bin/env python3
"""Prefetch pinned weights into the standard HF cache (Python 3.6 + curl).

The six entries below were verified against the pinned Hugging Face tree API.
LFS files use their recorded SHA256; JSON files were additionally downloaded and
checked against Git blob IDs before recording their SHA256 here.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
MODEL_ID = "ResembleAI/chatterbox"
FILES = [
    {"name": "t3_mtl23ls_v2.safetensors", "size": 2143989752,
     "sha256": "b1237586127ce98e7800a68e49938eb5092846862aabcb6e17b2fda7889a6c75"},
    {"name": "s3gen.pt", "size": 1057165844,
     "sha256": "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3"},
    {"name": "ve.pt", "size": 5698626,
     "sha256": "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1"},
    {"name": "conds.pt", "size": 107374,
     "sha256": "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e"},
    {"name": "Cangjie5_TC.json", "size": 1920163,
     "sha256": "7073fd9de919443ae88e0bd2449917a65fe54898a4413ed1edcc4b67f28bce8c",
     "git_blob": "d77891f84ca1db0d6f7058a4ee081d4bb0bfe88e"},
    {"name": "grapheme_mtl_merged_expanded_v1.json", "size": 69989,
     "sha256": "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf",
     "git_blob": "d27fb3f2fd38ca39b7bbfbf83a13b3c617e551df"},
]


def valid_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest() == expected["sha256"]


def model_url(expected):
    return "https://huggingface.co/{}/resolve/{}/{}?download=true".format(MODEL_ID, REVISION, expected["name"])


def _final_headers(raw):
    """Read only the final response; never print/store redirect Location URLs."""
    status, headers = None, {}
    for line in raw.decode("iso-8859-1").splitlines():
        matched = re.match(r"^HTTP/\S+\s+(\d{3})(?:\s|$)", line)
        if matched:
            status, headers = int(matched.group(1)), {}
        elif ":" in line and status is not None:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return status, headers


def _fetch_range(expected, directory, start, end):
    """Resume candidate bytes; only the final whole-model SHA accepts a model."""
    fragment = directory / "{:012d}-{:012d}.part".format(start, end)
    length = end - start + 1
    if fragment.exists():
        if fragment.stat().st_size != length:
            raise RuntimeError("Existing range fragment has the wrong length: " + fragment.name)
        return fragment
    staged = fragment.with_name(fragment.name + ".downloading")
    prefix_size = staged.stat().st_size if staged.exists() else 0
    if prefix_size > length:
        raise RuntimeError("Existing range prefix is longer than the requested range; preserving it")
    record_path = staged.with_name(staged.name + ".json")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    # Old downloaders did not save HTTP headers. Preserve those bytes as an
    # explicitly unverified candidate, exactly like the contiguous .incomplete
    # prefix. They cannot become an accepted model without the final full SHA.
    candidate_prefix = record.get("unverified_candidate_prefix_bytes", prefix_size)

    def save_record():
        checksum = hashlib.sha256()
        with staged.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                checksum.update(chunk)
        record = {"model_revision": REVISION, "model_sha256": expected["sha256"],
                  "range_start": start, "range_end": end,
                  "bytes": staged.stat().st_size, "prefix_sha256": checksum.hexdigest(),
                  "unverified_candidate_prefix_bytes": candidate_prefix,
                  "requires_full_model_sha256": True}
        temporary_record = record_path.with_name(record_path.name + ".tmp")
        temporary_record.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(temporary_record), str(record_path))

    if prefix_size:
        save_record()
    if prefix_size == length:
        os.replace(str(staged), str(fragment))
        return fragment
    request_start = start + prefix_size
    remaining = end - request_start + 1
    # Keep any earlier unknown-response tail intact. A unique file also avoids
    # overwriting an interrupted tail whose headers were never observed.
    tail = fragment.with_name(fragment.name + ".tail-" + uuid.uuid4().hex)
    response = subprocess.run([
        "curl", "--http1.1", "--fail", "--location", "--silent", "--show-error",
        "--proto", "=https", "--proto-redir", "=https",
        "--retry", "3", "--retry-delay", "2", "--connect-timeout", "30",
        "--max-time", "3600", "--speed-limit", "1024", "--speed-time", "120",
        "--range", "{}-{}".format(request_start, end), "--max-filesize", str(remaining),
        "--header", "Accept-Encoding: identity", "--dump-header", "-",
        "--output", str(tail), model_url(expected),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    status, headers = _final_headers(response.stdout)
    wanted_range = "bytes {}-{}/{}".format(request_start, end, expected["size"])
    if status != 206 or headers.get("content-range") != wanted_range:
        raise RuntimeError("Server did not honor the exact range for {} bytes {}-{}; tail preserved without joining".format(
            expected["name"], request_start, end))
    if headers.get("content-encoding", "identity").lower() != "identity":
        raise RuntimeError("Encoded range response cannot be assembled safely")
    if "content-length" in headers and headers["content-length"] != str(remaining):
        raise RuntimeError("Range Content-Length mismatch for " + expected["name"])
    received = tail.stat().st_size if tail.is_file() else 0
    if received > remaining or not received or (not response.returncode and received != remaining):
        raise RuntimeError("Range body length mismatch for " + expected["name"])
    # A failed transfer can still leave a useful prefix, but only after its
    # final headers proved the exact requested 206 range and identity encoding.
    if prefix_size:
        joined = staged.with_name(staged.name + ".joining")
        with joined.open("wb") as output:
            for source_path in (staged, tail):
                with source_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(str(joined), str(staged))
        tail.unlink()
    else:
        os.replace(str(tail), str(staged))
    save_record()
    if response.returncode:
        # Never log curl's stderr or redirect headers containing signed URLs.
        raise RuntimeError("Range transfer interrupted for {} (curl exit {}); preserved {} candidate bytes".format(
            expected["name"], response.returncode, staged.stat().st_size))
    if staged.stat().st_size != length:
        raise RuntimeError("Assembled range length mismatch for " + expected["name"])
    os.replace(str(staged), str(fragment))
    return fragment


def download_ranges(partial, expected, connections, range_bytes):
    """Keep the contiguous prefix and resume independent verified fragments."""
    prefix_size = partial.stat().st_size if partial.exists() else 0
    directory = partial.with_name(partial.name + ".ranges")
    directory.mkdir(exist_ok=True)
    # Global chunk boundaries allow later resumes to reuse complete fragments
    # even if another earlier attempt grew the contiguous prefix.
    ranges = []
    offset = prefix_size
    while offset < expected["size"]:
        end = min(((offset // range_bytes) + 1) * range_bytes, expected["size"]) - 1
        ranges.append((offset, end))
        offset = end + 1
    print("Range download {}: preserving {} prefix bytes, {} fragments, {} connections".format(
        expected["name"], prefix_size, len(ranges), connections), flush=True)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(_fetch_range, expected, directory, start, end) for start, end in ranges]
        fragments = [future.result() for future in futures]
    current_prefix = partial.stat().st_size if partial.exists() else 0
    if current_prefix != prefix_size:
        raise RuntimeError("Contiguous prefix changed during range download; stop other downloaders and resume")
    assembled = partial.with_name(partial.name + ".assembled")
    with assembled.open("wb") as destination:
        for source in ([partial] if prefix_size else []) + fragments:
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    if not valid_file(assembled, expected):
        raise RuntimeError("Assembled ranges failed full size/SHA256 validation: " + expected["name"])
    os.replace(str(assembled), str(partial))
    # Keep range fragments until final promotion. Their sizes allow a retry to
    # reuse completed work; the full SHA256 remains the acceptance authority.
    return fragments


def fetch_file(model_cache, expected, verify_only, connections=1, range_bytes=64 * 1024 * 1024):
    etag = expected.get("git_blob", expected["sha256"])
    blob = model_cache / "blobs" / etag
    partial = blob.with_name(blob.name + ".incomplete")
    target = model_cache / "snapshots" / REVISION / expected["name"]
    if not valid_file(blob, expected):
        if verify_only:
            raise RuntimeError("Missing or invalid cached file: " + expected["name"])
        if blob.exists():
            raise RuntimeError("Existing cache checksum mismatch; inspect before replacing: " + str(blob))
        # A finished but not-yet-promoted partial needs no new HTTP request.
        if not valid_file(partial, expected):
            if partial.exists() and partial.stat().st_size >= expected["size"]:
                raise RuntimeError("Partial file checksum mismatch; inspect before restarting: " + str(partial))
            print("Downloading {} ({} bytes; resumable)".format(expected["name"], expected["size"]), flush=True)
            if connections > 1 and expected["size"] > range_bytes:
                download_ranges(partial, expected, connections, range_bytes)
            else:
                subprocess.check_call([
                    "curl", "--fail", "--location", "--show-error", "--silent",
                    "--proto", "=https", "--proto-redir", "=https",
                    "--continue-at", "-", "--retry", "8", "--retry-delay", "2",
                    "--connect-timeout", "30", "--max-time", "43200",
                    "--speed-limit", "1024", "--speed-time", "120",
                    "--output", str(partial), model_url(expected),
                ])
        if not valid_file(partial, expected):
            raise RuntimeError("Downloaded file failed size/SHA256 validation: " + expected["name"])
        os.replace(str(partial), str(blob))
    if not verify_only:
        staged = target.with_name(target.name + ".prefetch-link")
        if staged.is_symlink():
            staged.unlink()
        os.symlink(os.path.relpath(str(blob), str(target.parent)), str(staged))
        os.replace(str(staged), str(target))
    if not target.is_file() or target.resolve() != blob.resolve():
        raise RuntimeError("Snapshot link is missing or incorrect: " + expected["name"])
    print("Verified {} {}".format(expected["name"], expected["sha256"]), flush=True)
    return dict(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-home", required=True, type=Path,
                        help="Host HF_HOME directory, e.g. ./cache/huggingface")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--connections-per-file", type=int, default=1,
                        help="Optional parallel byte ranges per large file; default 1 keeps ordinary curl resume")
    parser.add_argument("--range-mib", type=int, default=64,
                        help="Range fragment size in MiB (default 64)")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("--workers must be between 1 and 6")
    if not 1 <= args.connections_per_file <= 16 or not 1 <= args.range_mib <= 256:
        parser.error("--connections-per-file must be 1..16 and --range-mib must be 1..256")
    model_cache = args.cache_home / "hub" / "models--ResembleAI--chatterbox"
    (model_cache / "blobs").mkdir(parents=True, exist_ok=True)
    (model_cache / "snapshots" / REVISION).mkdir(parents=True, exist_ok=True)
    with (model_cache / ".prefetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_file, model_cache, item, args.verify_only,
                                   args.connections_per_file, args.range_mib * 1024 * 1024) for item in FILES]
            verified = [future.result() for future in futures]
        report = {"model_id": MODEL_ID, "revision": REVISION, "files": verified,
                  "verified_at": time.time(), "total_bytes": sum(item["size"] for item in FILES)}
        if not args.verify_only:
            staged = model_cache / "prefetch-report.json.tmp"
            staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            os.replace(str(staged), str(model_cache / "prefetch-report.json"))
    print(json.dumps({"status": "verified", "revision": REVISION,
                      "total_bytes": report["total_bytes"], "snapshot": str(model_cache / "snapshots" / REVISION)}), flush=True)


if __name__ == "__main__":
    main()
