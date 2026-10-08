"""Decode acoustic phones from a WAV without receiving source/reference text."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime-lib', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_lib.resolve()))
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    from transformers.models.wav2vec2_phoneme.tokenization_wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer

    model_id = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
    revision = 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
    # Model initialization also uses Torch's CPU pool. Bound it before
    # loading weights, rather than only after the expensive initialization.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    digest = hashlib.sha256(args.audio.read_bytes()).hexdigest()
    options = {'revision': revision, 'local_files_only': True}
    feature = Wav2Vec2FeatureExtractor.from_pretrained(model_id, **options)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_id, do_phonemize=False, **options)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, use_safetensors=False, **options)
    model.eval()
    decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(args.audio),
                                       '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'])
    samples = np.frombuffer(decoded, dtype='<f4').copy()
    inputs = feature(samples, sampling_rate=16000, return_tensors='pt')
    with torch.inference_mode():
        labels = torch.argmax(model(**inputs).logits, dim=-1)
    phones = tokenizer.batch_decode(labels)[0]
    if hashlib.sha256(args.audio.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Original waveform changed during phone observation')
    result = {'model': model_id, 'model_revision': revision, 'audio_sha256': digest,
              'audio_path': str(args.audio.resolve()), 'phones': phones, 'sample_rate': 16000,
              'conditioned_on_reference_text': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
