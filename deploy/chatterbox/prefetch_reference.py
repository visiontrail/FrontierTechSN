"""Fetch and verify the fixed official Space English reference (Python 3.6+)."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

VOICE_ID = "space-en-f1"
SPACE_REVISION = "c612a94210c0af895d6b649850d15b7ccca36594"
GENERATION = "1755807478904629"
SOURCE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac?generation=" + GENERATION
SIZE_BYTES = 83323
SHA256 = "e81950378b53827aed08d164ebd332a7b8c5805b8a97b4f79ab057071e359c72"


def reference_source():
    return {
        "kind": "official_space_reference", "space_revision": SPACE_REVISION,
        "url": SOURCE_URL, "generation": GENERATION,
        "filename": "en_f1.flac", "size_bytes": SIZE_BYTES,
    }


def verify_reference(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != SIZE_BYTES:
        raise RuntimeError("Official Space English reference is missing or has the wrong size")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != SHA256:
        raise RuntimeError("Official Space English reference checksum mismatch")
    return SHA256


def prepare_reference(destination):
    """Preserve existing files; only promote a fully verified original FLAC."""
    destination = Path(destination)
    if destination.exists():
        verify_reference(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".en_f1-", suffix=".incomplete", dir=str(destination.parent), delete=False,
    ) as output:
        temporary = Path(output.name)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            if response.headers.get("x-goog-generation") != GENERATION:
                raise RuntimeError("Unexpected official reference generation")
            received = 0
            while True:
                block = response.read(min(65536, SIZE_BYTES + 1 - received))
                if not block:
                    break
                received += len(block)
                if received > SIZE_BYTES:
                    raise RuntimeError("Official reference download exceeds its pinned size")
                output.write(block)
        output.flush()
        os.fsync(output.fileno())
    verify_reference(temporary)
    try:
        # Do not overwrite a file another process installed while downloading.
        os.link(str(temporary), str(destination))
    except FileExistsError:
        verify_reference(destination)
    temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="voices/en_f1.flac")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_reference(args.output)
    else:
        prepare_reference(args.output)
    print("Verified official Space English reference: {} bytes, sha256 {}".format(SIZE_BYTES, SHA256))


if __name__ == "__main__":
    main()
