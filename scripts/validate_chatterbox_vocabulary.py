#!/usr/bin/env python3
"""Resume an isolated, source-bound Chatterbox common-vocabulary acoustic audit.

The corpus is verification data, never a production pronunciation allowlist.
Uses the installed voice, ordinary provider parameters and existing bounded
retry/checkpoint behavior. Does not create application tasks or publish media.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline import chatterbox, tts, tts_pronunciation as pronunciation  # noqa: E402
from backend.runtime import RUNTIME_IDENTITY  # noqa: E402
from scripts.validate_chatterbox import TERMINAL, require_idle  # noqa: E402

FIXTURE = ROOT / 'backend/tests/fixtures/chatterbox_common_vocabulary.json'
DEFAULT_OUTPUT = ROOT / 'outputs/chatterbox-uppercase-20261006/full-corpus'
SECONDARY_MODEL = 'mlx-community/whisper-small-mlx'
WORD_ALIASES = {'sql': [['sequel']], 'saas': [['sass']], 'iii': [['3']]}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    staged.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def corpus(fixture: dict, observed: dict | None = None) -> list[dict]:
    rows, seen = [], set()

    def add(term, kind, category, *, text=None, identifier=None, actions=None):
        if identifier is None and term in seen:
            raise ValueError(f'Duplicate base vocabulary entry: {term}')
        if identifier is None:
            seen.add(term)
        text = text or fixture['overrides'].get(term) or f'The report mentions {term} today.'
        if actions is None:
            actions = ({c['source']: kind for c in pronunciation.candidates(text)}
                       if kind in {'spell', 'keep'} else {})
            if term in fixture.get('mode_overrides', {}):
                actions = {c['source']: fixture['mode_overrides'][term] for c in pronunciation.candidates(text)}
        rows.append({'id': identifier or 'base-' + digest(term)[:12], 'term': term,
                     'text': text, 'category': category, 'expected_actions': actions})

    for key, kind in [('initialisms', 'spell'), ('word_readings', 'keep')]:
        for category, terms in fixture[key].items():
            for term in terms.split():
                add(term, kind, category)
    for key in ['contextual_readings', 'mixed_case_and_symbols']:
        for term in fixture[key].split():
            add(term, 'contextual', key)
    for row in fixture['context_cases']:
        add(row['term'], 'contextual', 'context_contrast', text=row['text'],
            identifier=row['id'], actions=row['expected_actions'])
    for key, suffix, text in [
        ('plural_bases', 's', 'Several {term} appear in the report.'),
        ('possessive_bases', "'s", 'The {term} result is available.'),
    ]:
        for base in fixture[key].split():
            term = base + suffix
            kind = 'keep' if base in fixture['word_readings']['acronyms'].split() else 'contextual'
            add(term, kind, key, text=text.format(term=term), identifier=key + '-' + digest(term)[:12])
    for key in ['dotted_forms', 'product_forms']:
        for term in fixture[key].split():
            add(term, 'contextual', key, identifier=key + '-' + digest(term)[:12])
    for term in sorted((observed or {}).get('terms', {})):
        if term in seen:
            continue
        add(term, 'contextual', 'observed_news', text=(
            'The study reaches Phase III.' if term == 'III' else None
        ), actions={'III': 'keep'} if term == 'III' else None)
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Corpus case IDs must be unique')
    return rows


def prepare(directory: Path, fixture: dict, observed: dict | None) -> dict:
    cases = corpus(fixture, observed)
    identity = {'version': 1, 'cases': cases, 'fixture_sha256': digest(fixture),
                'observed_sha256': digest(observed) if observed else None,
                'voice': 'space-en-f1', 'language': 'en',
                'secondary_model': SECONDARY_MODEL, 'cache_contract': chatterbox.cache_contract(),
                'batch_size': 12}
    coverage = {'corpus_sha256': digest(identity), 'case_count': len(cases),
                'term_count': len({r['term'] for r in cases}),
                'uppercase_letters': ''.join(sorted({c for r in cases for c in r['term'] if 'A' <= c <= 'Z'})),
                'batch_count': (len(cases) + 11) // 12}
    marker = directory / 'corpus.json'
    if marker.exists():
        saved = read(marker)
        if saved['identity'] != identity:
            raise ValueError('Frozen corpus or provider contract changed; preserve this run and use another output directory')
        if any(saved.get(key) != value for key, value in coverage.items()):
            raise ValueError('Frozen corpus coverage/hash metadata does not match its complete case list')
        return saved
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Output belongs to another run without a corpus marker')
    result = {'identity': identity, 'created_at': datetime.now(timezone.utc).isoformat(), **coverage}
    write(marker, result)
    return result


def mode_report(text: str, cases: list[dict], plan: dict) -> dict:
    failures, offset = [], 0
    for case in cases:
        start = text.index(case['text'], offset)
        offset = start + len(case['text'])
        for decision, candidate in zip(plan['decisions'], pronunciation.candidates(text), strict=True):
            if not start <= candidate['start'] < offset:
                continue
            expected = case['expected_actions'].get(decision['source'])
            allowed = [expected] if isinstance(expected, str) else expected
            if allowed and decision['action'] not in allowed:
                failures.append({'case_id': case['id'], 'source': decision['source'],
                                 'expected_action': expected, 'actual_decision': decision})
    return {'passed': not failures, 'failures': failures}


def all_target_report(text: str, words: list[dict]) -> dict:
    """Audit kept words and mixed-case tokens as well as selected spell edits.

    Preserve raw observations. Only documented word readings at an exact
    source-aligned replacement can normalize to their source representation.
    No arbitrary brand-prefix alias, missing letter or repeated word is erased.
    """
    observed, indexes = tts._transcript_tokens(words)
    # Preserve an actual written ampersand for the production guard's
    # source-bound AIR &D repartition, rather than inventing an ASR "and".
    observed = ['&' if token == 'and' and '&' in str(words[index].get('text', '')) else token
                for token, index in zip(observed, indexes, strict=True)]
    expected = tts._lexical_tokens(text)
    replacements, aliases = {}, []
    for tag, start, end, left, right in SequenceMatcher(a=expected, b=observed, autojunk=False).get_opcodes():
        if tag == 'replace' and end == start + 1 and observed[left:right] in WORD_ALIASES.get(expected[start], []):
            replacements[left] = (right, expected[start:end])
            aliases.append({'source': expected[start], 'observed': observed[left:right], 'source_index': start})
    normalized, cursor = [], 0
    while cursor < len(observed):
        if cursor in replacements:
            cursor, tokens = replacements[cursor]
            normalized.extend(tokens)
        else:
            normalized.append(observed[cursor])
            cursor += 1
    probes = []
    symbols = list(chatterbox._symbol_initialisms(text))
    compounds = [m for m in re.finditer(r'\S+', text)
                 if sum('A' <= c <= 'Z' for c in m[0]) >= 2 or re.search(r'\d', m[0])]
    for match in chatterbox.PRONUNCIATION_TOKEN_RE.finditer(text):
        if any(s.start() < match.end() and s.end() > match.start() for s in symbols):
            continue
        has_number = bool(re.search(r'\d', match[0]))
        if not any(m.start() <= match.start() and match.end() <= m.end() for m in compounds):
            continue
        # Keep numeric/decimal representation separate from the name's
        # alphabetic parts, including lowercase product suffixes such as 4o.
        if has_number:
            probes.extend({'source': m[0], 'start': match.start() + m.start(),
                           'end': match.start() + m.end()}
                          for m in re.finditer(r'[A-Za-z]+', match[0]))
        else:
            probes.append({'source': match[0], 'start': match.start(), 'end': match.end()})
    report = chatterbox._pronunciation_report(text, [{'text': ' '.join(normalized)}], probes)
    # Model suffixes must retain their values, order and multiplicity too.
    def numbers(tokens):
        result = []
        for value in tokens:
            if re.fullmatch(r'decimalnumber\d+point\d+', value):
                result.append(value)
            else:
                result.extend(re.findall(r'\d+', value))
        return result
    numeric = {'expected': numbers(expected), 'observed': numbers(normalized)}
    numeric['passed'] = numeric['expected'] == numeric['observed']
    operators = {'required': 'C++' in text or 'C#' in text, 'passed': True}
    if operators['required']:
        # C# can be transcribed as either C# or its conventional name C sharp.
        alternatives = [expected, ['sharp' if t == 'hashsymbol' else t for t in expected]]
        operators['passed'] = normalized in alternatives
        operators.update(expected_alternatives=alternatives, observed=normalized)
    multiplicity = []
    for canonical, forms in WORD_ALIASES.items():
        if canonical not in expected:
            continue
        values = {canonical, *(form[0] for form in forms if len(form) == 1)}
        count = lambda tokens: sum(token in values for token in tokens)
        multiplicity.append({'source': canonical, 'expected_count': count(expected),
                             'observed_count': count(normalized), 'passed': count(expected) == count(normalized)})
    report.update(raw_transcript=tts._raw_transcript(words), source_bound_word_aliases=aliases,
                  word_reading_multiplicity=multiplicity,
                  numeric_suffixes=numeric, programming_symbols=operators)
    report['passed'] = report['passed'] and numeric['passed'] and operators['passed'] and all(r['passed'] for r in multiplicity)
    return report


async def independent_part_audit(wav: Path, text: str, edits: list[dict], *, directory: Path, plan: dict) -> dict:
    """Disprove a part before acceptance, using the same remaining job attempts."""
    identity = {'audio_sha256': sha(wav), 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'plan_sha256': plan['plan_sha256'], 'model': SECONDARY_MODEL}
    target = directory / 'independent-parts' / identity['source_sha256'] / identity['audio_sha256']
    target.mkdir(parents=True, exist_ok=True)
    transcript = target / 'transcript.json'
    marker = target / 'identity.json'
    if not marker.exists() or read(marker) != identity or not transcript.exists():
        env = os.environ.copy()
        env['HF_HUB_OFFLINE'] = '1'
        done = await asyncio.to_thread(subprocess.run, [sys.executable, '-m', 'backend.pipeline.av_sync',
            'transcribe-mlx', str(wav), str(transcript), '--model', SECONDARY_MODEL, '--language', 'en'],
            cwd=ROOT, env=env, capture_output=True, timeout=180)
        if done.returncode:
            raise RuntimeError('Independent recognizer unavailable: ' + done.stderr.decode()[-1000:])
        write(marker, identity)
    words = json.loads(transcript.read_text())
    spelling = chatterbox._pronunciation_report(text, words, edits)
    report = {'identity': identity, 'transcript_sha256': sha(transcript),
              'targets': all_target_report(text, words), 'spelling': spelling,
              'tail': chatterbox._trailing_speech_report(text, words),
              'normalized_tail': chatterbox._trailing_speech_report(text, [{'text': spelling['normalized_transcript']}])}
    if sha(wav) != identity['audio_sha256']:
        raise RuntimeError('Waveform changed during its independent audit')
    report['passed'] = all(report[key]['passed'] for key in ['targets', 'spelling', 'tail', 'normalized_tail'])
    write(target / 'audit.json', report)
    if not report['passed']:
        raise tts.TtsIntegrityError('Independent vocabulary audit rejected speech: ' + json.dumps(report, ensure_ascii=False))
    return report


def composition_audio_window(tasks: list[dict], queue: dict) -> dict:
    """Allow private speech QA only after every active production voice is final."""
    active = [task for task in tasks if task['status'] not in TERMINAL]
    if queue.get('items'):
        raise RuntimeError('Queued production work must finish before private speech validation')
    owner = queue.get('active')
    ids = {task['id'] for task in active}
    if owner and owner.get('id') not in ids:
        raise RuntimeError('Shared queue owner is not an identified composing task')
    anchors = []
    for task in active:
        if task['status'] != 'composing':
            raise RuntimeError('Active production work has not finished its speech stage')
        wav = Path(task.get('audio_path') or '')
        manifest_path = wav.parent / 'tts_manifest.json'
        if not wav.is_file() or not manifest_path.is_file():
            raise RuntimeError('Composing task has no completed narration and manifest')
        manifest = read(manifest_path)
        integrity = manifest.get('integrity') or {}
        if (integrity.get('passed') is not True or integrity.get('required') is not True
                or integrity.get('accepted_source_coverage') != 1.0
                or integrity.get('accepted_source_words') != manifest.get('source_word_count')
                or manifest.get('chunk_count') != len(manifest.get('parts') or [])
                or manifest.get('output_audio_sha256') != sha(wav)):
            raise RuntimeError('Composing narration is incomplete or differs from its final waveform hash')
        tts._read_pcm_wav(wav)
        anchors.append({'task_id': task['id'], 'status': task['status'],
                        'audio_sha256': manifest['output_audio_sha256'],
                        'manifest_sha256': sha(manifest_path)})
    return {'available': True, 'production_audio_anchors': anchors,
            'queue_active': owner, 'queued_items': 0}


async def require_audio_window(app_url: str, *, allow_composing: bool = False) -> dict:
    if not allow_composing:
        return {'available': True, **await require_idle(app_url)}
    import httpx
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        responses = await asyncio.gather(*(client.get(app_url.rstrip('/') + '/api/' + endpoint)
                                          for endpoint in ['tasks', 'explainers/queue', 'runtime']))
    for response in responses:
        response.raise_for_status()
    report = composition_audio_window(responses[0].json()['tasks'], responses[1].json())
    report.update(checked_at=datetime.now(timezone.utc).isoformat(), application_runtime=responses[2].json())
    return report


async def audit_audio(directory: Path, source: str, plan: dict) -> dict:
    audio = directory / 'audio'
    reports, offset = [], 0
    for part in sorted(audio.glob('tts_input_part_*.txt')):
        text = part.read_text()
        start = source.index(text, offset)
        offset = start + len(text)
        edits = [{**e, 'start': e['start'] - start, 'end': e['end'] - start}
                 for e in plan['edits'] if start <= e['start'] and e['end'] <= offset]
        wav = part.with_name(part.stem + '_generated.wav')
        metadata = read(tts._part_metadata_path(wav))
        synthesis = chatterbox._render_provider_span(source, start, offset, plan['edits'])
        expected_identity = {
            'text': text, 'synthesis_text': synthesis, 'voice': 'space-en-f1', 'language': 'en',
            'pronunciation_plan_sha256': plan['plan_sha256'], 'synthesis_speed_ratio': 1.0,
        }
        if any(metadata['identity'].get(key) != value for key, value in expected_identity.items()):
            raise ValueError('Accepted part identity differs from the frozen source, plan or installed voice')
        if metadata['synthesis_request']['text'] != synthesis:
            raise ValueError('Accepted speech request differs from the source-bound provider input')
        if not chatterbox._cached_part(wav, metadata['identity'], verify_text=True):
            raise ValueError('Accepted production part no longer matches its waveform or identity')
        primary_words = json.loads((audio / 'verification' / part.stem / 'transcript.json').read_text())
        target = directory / 'independent' / part.stem
        transcript = target / 'transcript.json'
        identity = {'audio_sha256': sha(wav), 'source_sha256': sha(part),
                    'plan_sha256': plan['plan_sha256'], 'model': SECONDARY_MODEL}
        if not (target / 'identity.json').exists() or read(target / 'identity.json') != identity or not transcript.exists():
            target.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env['HF_HUB_OFFLINE'] = '1'
            done = await asyncio.to_thread(subprocess.run, [sys.executable, '-m', 'backend.pipeline.av_sync',
                'transcribe-mlx', str(wav), str(transcript), '--model', SECONDARY_MODEL, '--language', 'en'],
                cwd=ROOT, env=env, capture_output=True, timeout=180)
            if done.returncode:
                raise RuntimeError('Independent recognizer unavailable: ' + done.stderr.decode()[-1000:])
            write(target / 'identity.json', identity)
        secondary_words = json.loads(transcript.read_text())
        row = {'part': part.name, 'identity': identity,
               'primary': all_target_report(text, primary_words),
               'secondary': all_target_report(text, secondary_words),
               'secondary_spell_gate': chatterbox._pronunciation_report(text, secondary_words, edits),
               'secondary_tail_gate': chatterbox._trailing_speech_report(text, secondary_words)}
        row['secondary_normalized_tail_gate'] = chatterbox._trailing_speech_report(
            text, [{'text': row['secondary_spell_gate']['normalized_transcript']}])
        if sha(wav) != identity['audio_sha256']:
            raise ValueError('Accepted waveform changed during its final audit')
        row['passed'] = all(row[k]['passed'] for k in ['primary', 'secondary', 'secondary_spell_gate',
                                                     'secondary_tail_gate', 'secondary_normalized_tail_gate'])
        write(target / 'audit.json', row)
        reports.append(row)
        print(json.dumps({'part': part.name, 'dual_audit_passed': row['passed'],
                          'primary': row['primary']['raw_transcript'], 'secondary': row['secondary']['raw_transcript']}, ensure_ascii=False), flush=True)
    return {'passed': bool(reports) and all(r['passed'] for r in reports), 'parts': reports}


async def run(directory: Path, marker: dict, *, allow_composing: bool = False) -> None:
    rows = marker['identity']['cases']
    outcomes = []
    for index in range(marker['batch_count']):
        window = await require_audio_window('http://localhost:8101', allow_composing=allow_composing)
        cases = rows[index * 12:(index + 1) * 12]
        group = directory / f'batch-{index + 1:03d}'
        group.mkdir(exist_ok=True)
        write(group / 'execution-window.json', window)
        script = group / 'script.txt'
        source = '\n'.join(row['text'] for row in cases)
        if script.exists() and script.read_text() != source:
            raise ValueError('Batch source changed beside its checkpoints')
        script.write_text(source)
        normalized = pronunciation.normalize_text(source)
        plan = await pronunciation.prepare_pronunciation(normalized, group / 'audio',
            policy=marker['identity']['cache_contract']['pronunciation_policy'], emit=print)
        modes = mode_report(normalized, cases, plan)
        write(group / 'mode-audit.json', modes)
        outcome = {'batch': index + 1, 'case_ids': [r['id'] for r in cases],
                   'source_sha256': sha(script), 'plan_sha256': plan['plan_sha256'], 'mode_audit': modes}
        if not modes['passed']:
            outcome.update(passed=False, status='mode_mismatch_before_synthesis')
        else:
            async def audit_part(wav, text, edits):
                return await independent_part_audit(wav, text, edits, directory=group, plan=plan)
            async def before_submit(context):
                checked = await require_audio_window('http://localhost:8101', allow_composing=allow_composing)
                checked['request_context'] = context
                path = group / 'synthesis-windows.json'
                previous = json.loads(path.read_text()) if path.exists() else []
                if not isinstance(previous, list):
                    raise ValueError('Synthesis-window archive is invalid; preserving it')
                write(path, [*previous, checked])
                return checked
            try:
                result = await chatterbox._generate_chatterbox(str(script), str(group / 'audio'),
                    'space-en-f1', 'en', log=print, emit=print, verify_text=True,
                    part_auditor=audit_part, audit_all_parts=True, before_submit=before_submit)
            except tts.TtsIntegrityError as error:
                # Only positive integrity rejection advances; transport or
                # observation uncertainty preserves this batch and stops here.
                outcome.update(passed=False, status='bounded_integrity_rejection', error=str(error))
            else:
                audit = await audit_audio(group, normalized, plan)
                outcome.update(passed=audit['passed'], status='audited', audio_path=result,
                               audio_sha256=sha(Path(result)), audit=audit)
        write(group / 'outcome.json', outcome)
        outcomes.append(outcome)
        complete = len(outcomes) == marker['batch_count']
        audited = complete and all(r['status'] == 'audited' for r in outcomes)
        summary = {'corpus_sha256': marker['corpus_sha256'], 'case_count': marker['case_count'],
                   'term_count': marker['term_count'], 'completed_batches': len(outcomes),
                   'total_batches': marker['batch_count'], 'all_batches_attempted': complete,
                   'all_batches_audited': audited,
                   'passed': complete and all(r['passed'] for r in outcomes),
                   'runtime_identity': RUNTIME_IDENTITY, 'outcomes': outcomes}
        write(directory / 'summary.json', summary)
        print('VOCABULARY BATCH', index + 1, '/', marker['batch_count'], outcome['status'], outcome['passed'], flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--observed', type=Path)
    parser.add_argument('--allow-composing', action='store_true',
                        help='Use an identified composition window with hash-verified finished production narration')
    args = parser.parse_args()
    directory = args.output.expanduser().resolve()
    base = (ROOT / 'outputs').resolve()
    if directory == base or not directory.is_relative_to(base):
        parser.error('Use an owned child directory under outputs')
    marker = prepare(directory, read(FIXTURE), read(args.observed) if args.observed else None)
    print(json.dumps({k:marker[k] for k in ['corpus_sha256','case_count','term_count','batch_count','uppercase_letters']}), flush=True)
    if args.action == 'run':
        asyncio.run(run(directory, marker, allow_composing=args.allow_composing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
