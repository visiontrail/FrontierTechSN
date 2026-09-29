#!/usr/bin/env python3
"""Install the pinned Chinese segmenter cache using Python 3.6 and curl.

Source: spacy-pkuseg 1.0.1's official wheel, spacy_pkuseg/config.py,
model_urls/model_hash['spacy_ontonotes']. Size comes from the official release:
https://api.github.com/repos/explosion/spacy-pkuseg/releases/tags/v0.0.26
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import uuid
import zipfile

MODEL_NAME = "spacy_ontonotes"
ARCHIVE_NAME = MODEL_NAME + ".zip"
ARCHIVE_URL = "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip"
ARCHIVE_SHA256 = "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe"
ARCHIVE_SIZE = 34567143


def stream_hash(stream):
    checksum = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def valid_archive(path):
    if not path.is_file() or path.stat().st_size != ARCHIVE_SIZE:
        return False
    with path.open("rb") as stream:
        return stream_hash(stream) == ARCHIVE_SHA256


def safe_members(archive):
    members = archive.infolist()
    names = set()
    for member in members:
        name = member.filename
        parts = PurePosixPath(name).parts
        mode = (member.external_attr >> 16) & 0xFFFF
        if (not name or not parts or name.startswith(("/", "\\")) or "\\" in name or
                ".." in parts or ":" in parts[0] or stat.S_ISLNK(mode) or
                (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)) or name in names):
            raise RuntimeError("Unsafe or duplicate ZIP member")
        names.add(name)
    return members


def _verify_extracted(archive, members, target):
    if not target.is_dir() or target.is_symlink():
        raise RuntimeError("PKUSEG extracted model is missing or is a symlink")
    hashes = {}
    for member in members:
        destination = target / member.filename
        if os.path.commonpath([str(target.resolve()), str(destination.resolve())]) != str(target.resolve()):
            raise RuntimeError("PKUSEG extracted path escapes the model directory")
        if member.is_dir():
            if not destination.is_dir() or destination.is_symlink():
                raise RuntimeError("PKUSEG extracted directory is missing or changed")
            continue
        if (not destination.is_file() or destination.is_symlink() or
                destination.stat().st_size != member.file_size):
            raise RuntimeError("PKUSEG extracted file is missing or changed: " + member.filename)
        with archive.open(member) as source:
            expected_hash = stream_hash(source)
        with destination.open("rb") as actual:
            if stream_hash(actual) != expected_hash:
                raise RuntimeError("PKUSEG extracted file checksum mismatch: " + member.filename)
        hashes[member.filename] = expected_hash
    return hashes


def verify_cache(cache_home):
    """Read-only verification before model load; never downloads or repairs."""
    cache_home = Path(cache_home)
    archive_path = cache_home / ARCHIVE_NAME
    if not valid_archive(archive_path):
        raise RuntimeError("Pinned PKUSEG archive missing or invalid; run prefetch_pkuseg.py first")
    with zipfile.ZipFile(str(archive_path)) as archive:
        hashes = _verify_extracted(archive, safe_members(archive), cache_home / MODEL_NAME)
    return {"model": MODEL_NAME, "archive_sha256": ARCHIVE_SHA256,
            "archive_size": ARCHIVE_SIZE, "files": hashes}


def require_cached_download(url, model_dir, hash_prefix, progress=True):
    """Replacement for PKUSEG's downloader in the inference-only process."""
    if url != ARCHIVE_URL or hash_prefix != ARCHIVE_SHA256:
        raise RuntimeError("Unpinned PKUSEG model request is prohibited")
    verify_cache(Path(model_dir))


def prepare_cache(cache_home, verify_only=False):
    cache_home = Path(cache_home)
    cache_home.mkdir(parents=True, exist_ok=True)
    with (cache_home / ".prefetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verify_only:
            return verify_cache(cache_home)
        archive_path = cache_home / ARCHIVE_NAME
        partial = cache_home / (ARCHIVE_NAME + ".incomplete")
        if not valid_archive(archive_path):
            if archive_path.exists():
                raise RuntimeError("Existing PKUSEG archive is invalid; preserving it for inspection")
            if not valid_archive(partial):
                if partial.exists() and partial.stat().st_size >= ARCHIVE_SIZE:
                    raise RuntimeError("Existing PKUSEG partial is invalid; preserving it for inspection")
                subprocess.check_call([
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--proto", "=https", "--proto-redir", "=https", "--continue-at", "-",
                    "--retry", "8", "--retry-delay", "2", "--connect-timeout", "30",
                    "--max-time", "3600", "--output", str(partial), ARCHIVE_URL,
                ])
            if not valid_archive(partial):
                raise RuntimeError("Downloaded PKUSEG archive failed exact size/SHA256 validation")
            os.replace(str(partial), str(archive_path))
        target = cache_home / MODEL_NAME
        if not target.exists():
            staged = cache_home / (".pkuseg-extract-" + uuid.uuid4().hex)
            staged.mkdir()
            try:
                with zipfile.ZipFile(str(archive_path)) as archive:
                    members = safe_members(archive)
                    for member in members:
                        destination = staged / member.filename
                        if member.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(member) as source, destination.open("wb") as output:
                                shutil.copyfileobj(source, output)
                    _verify_extracted(archive, members, staged)
                os.replace(str(staged), str(target))
            finally:
                if staged.exists():
                    shutil.rmtree(str(staged))
        # Existing valid archives/models are reused. Unknown modifications fail
        # instead of silently replacing user-owned or interrupted cache state.
        result = verify_cache(cache_home)
        report = cache_home / "prefetch-report.json.tmp"
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(str(report), str(cache_home / "prefetch-report.json"))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-home", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_cache(args.cache_home, args.verify_only), sort_keys=True))


if __name__ == "__main__":
    main()
