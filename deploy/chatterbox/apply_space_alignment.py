"""Apply a narrow, hash-checked Space V2 stopping patch to the pinned package."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

SPACE_REVISION = "c612a94210c0af895d6b649850d15b7ccca36594"
ORIGINAL_T3_SHA256 = "bfb689c9018a329aa3a1a364aaa6d03a68a97b7352cbc394a13d4711742f9a0a"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patched_source(source):
    if sha256(source.encode()) != ORIGINAL_T3_SHA256:
        raise RuntimeError("Unexpected upstream T3 source; alignment patch refused")
    replacements = [
        ("from ..utils import AttrDict", "from ..utils import AttrDict\nfrom space_alignment import AlignmentStreamAnalyzer, alignment_guard"),
        ("    @torch.inference_mode()\n    def inference(", "    @torch.inference_mode()\n    @alignment_guard\n    def inference("),
        ("        # ---- Initial Forward Pass (no kv_cache yet) ----", """        # Restore the fixed multilingual Space's alignment/EOS mechanism.
        if self.hp.is_multilingual:
            self._space_alignment = AlignmentStreamAnalyzer(
                self.tfmr, None,
                text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                alignment_layer_idx=9, eos_idx=self.hp.stop_speech_token,
            )

        # ---- Initial Forward Pass (no kv_cache yet) ----"""),
        ("            # Apply repetition penalty\n", """            # Same placement as Space: after CFG, before sampling processors.
            if self._space_alignment is not None:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = self._space_alignment.step(logits, next_token=last_token)

            # Apply repetition penalty
"""),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Unexpected T3 patch anchor count")
        source = source.replace(old, new)
    start = source.index("    def inference(")
    end = source.index("    def inference_turbo(", start)
    inference = source[start:end]
    if inference.count("output_attentions=False") != 2:
        raise RuntimeError("Unexpected T3 attention call count")
    source = source[:start] + inference.replace("output_attentions=False", "output_attentions=True") + source[end:]
    compile(source, "patched_t3.py", "exec")
    return source


def package_path():
    return Path(importlib.metadata.distribution("chatterbox-tts").locate_file("chatterbox/models/t3/t3.py"))


def verify_patch():
    base = Path(__file__).parent
    manifest = json.loads((base / "alignment_patch_manifest.json").read_text())
    if (manifest["space_revision"] != SPACE_REVISION
            or manifest["original_t3_sha256"] != ORIGINAL_T3_SHA256
            or manifest["patched_t3_sha256"] != sha256(package_path().read_bytes())
            or manifest["alignment_sha256"] != sha256((base / "space_alignment.py").read_bytes())
            or manifest["patcher_sha256"] != sha256(Path(__file__).read_bytes())):
        raise RuntimeError("Space alignment installation checksum mismatch")
    return manifest


def main():
    base = Path(__file__).parent
    target = package_path()
    patched = patched_source(target.read_text())
    target.write_text(patched)
    manifest = {
        "space_revision": SPACE_REVISION,
        "original_t3_sha256": ORIGINAL_T3_SHA256,
        "patched_t3_sha256": sha256(patched.encode()),
        "alignment_sha256": sha256((base / "space_alignment.py").read_bytes()),
        "patcher_sha256": sha256(Path(__file__).read_bytes()),
        "attention_implementation": "eager",
    }
    (base / "alignment_patch_manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    verify_patch()
    print("Fixed Space alignment patch verified")


if __name__ == "__main__":
    main()
