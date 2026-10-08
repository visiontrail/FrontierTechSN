"""Independent checks for extra speech in English pronunciation chunks."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from backend import config

MODEL = 'mlx-community/whisper-small-mlx'
AUDIT_VERSION = 1


async def audit_part(wav: Path, text: str, edits: list[dict], directory: Path) -> dict:
    from backend.pipeline import chatterbox, tts, tts_pronunciation

    scope = 'independent_tail_and_extra_initialism_check'
    if not tts_pronunciation.candidates(text) and not chatterbox._symbol_initialisms(text):
        return {'required': False, 'passed': True, 'scope': scope}
    identity = {
        'audio_sha256': tts._file_sha256(wav),
        'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'edits_sha256': chatterbox._digest(edits),
        'model': MODEL, 'version': AUDIT_VERSION,
    }
    target = directory / identity['source_sha256'] / identity['audio_sha256']
    target.mkdir(parents=True, exist_ok=True)
    transcript, marker = target / 'transcript.json', target / 'identity.json'
    binding = chatterbox._read_json(marker)
    if (not transcript.exists() or binding.get('identity') != identity
            or binding.get('transcript_sha256') != tts._file_sha256(transcript)):
        env = os.environ.copy()
        env['HF_HUB_OFFLINE'] = '1'
        try:
            done = await asyncio.to_thread(subprocess.run, [
                sys.executable, '-m', 'backend.pipeline.av_sync', 'transcribe-mlx',
                str(wav), str(transcript), '--model', MODEL, '--language', 'en',
            ], cwd=config.PROJECT_ROOT, env=env, capture_output=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError('Independent recognizer unavailable; preserving the completed job') from error
        if done.returncode or not transcript.exists():
            raise RuntimeError('Independent recognizer unavailable; preserving the completed job')
        chatterbox._write_json(marker, {
            'identity': identity, 'transcript_sha256': tts._file_sha256(transcript),
        })
    try:
        words = json.loads(transcript.read_text())
        if (not isinstance(words, list) or not words
                or any(not isinstance(word, dict) for word in words)
                or not tts._raw_transcript(words).strip()):
            raise ValueError('Independent recognizer did not return word observations')
        spelling = chatterbox._pronunciation_report(text, words, edits)
    except (OSError, ValueError, TypeError) as error:
        chatterbox._write_json(marker, {'observation_unavailable': True, 'identity_before_failure': identity})
        raise RuntimeError('Independent observations unavailable; preserving the completed job') from error
    extra_initialisms, identifier_mismatches = [], []
    for difference in spelling['differences']:
        if (any(re.search(r'\d', token) for token in difference['source'])
                or (not difference['source'] and any(
                    re.search(r'\d', token) for token in tts._canonicalize_number_tokens(difference['asr'])))):
            identifier_mismatches.append(difference)
        for form in set(spelling['targets']):
            def count(tokens):
                total, cursor = 0, 0
                while cursor < len(tokens):
                    end = chatterbox._match_spoken_letters(tokens, cursor, len(tokens), form)
                    if end is None:
                        cursor += 1
                    else:
                        total += 1
                        cursor = end
                return total
            if count(difference['asr']) > count(difference['source']):
                extra_initialisms.append({'target': form, 'difference': difference})
    report = {
        'required': True, 'scope': scope, 'identity': identity,
        'transcript_sha256': tts._file_sha256(transcript),
        'raw_transcript': tts._raw_transcript(words),
        'tail': chatterbox._trailing_speech_report(text, words),
        'normalized_tail': chatterbox._trailing_speech_report(
            text, [{'text': spelling['normalized_transcript']}]),
        'extra_initialisms': extra_initialisms,
        'identifier_mismatches': identifier_mismatches,
    }
    if tts._file_sha256(wav) != identity['audio_sha256']:
        raise RuntimeError('Original waveform changed during independent verification')
    report['passed'] = (report['tail']['passed'] and report['normalized_tail']['passed']
                        and not extra_initialisms and not identifier_mismatches)
    chatterbox._write_json(target / 'audit.json', report)
    if not report['passed']:
        raise tts.TtsIntegrityError('Independent Chatterbox check found unrequested speech: ' +
                                  json.dumps({'tail': report['tail']['extra_words'],
                                              'extra_initialisms': extra_initialisms,
                                              'identifier_mismatches': identifier_mismatches}))
    return report
