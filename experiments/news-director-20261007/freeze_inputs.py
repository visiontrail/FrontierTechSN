"""Copy only required inputs from a live task and prove each copy stayed stable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(); out = args.out.resolve()
    if (out / 'input-manifest.json').exists():
        raise SystemExit('Snapshot already exists. Reuse it or choose a different output directory.')
    files = ['script.txt','summary.json','storyboard.json','transcript.json',
             'review/fact_check_report.json','footage/manifest.json',
             'audio/tts_input_generated.wav','audio/paced_narration.wav',
             'audio/program_mix.wav','audio/program_pacing_report.json',
             'audio/music_mix_report.json']
    clips = json.loads((source / 'footage/manifest.json').read_text())['clips']
    files += [clip['local_path'] for clip in clips]
    records = []
    for name in files:
        src = (source / name).resolve()
        if not src.is_relative_to(source):
            raise ValueError(f'Input path escapes source task: {name}')
        dst = out / 'inputs' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(src)
        shutil.copy2(src, dst)
        if not (before == sha256(dst) == sha256(src)):
            raise RuntimeError(f'Input changed while copying: {name}. No snapshot accepted.')
        records.append({'path':name, 'sha256':before, 'bytes':dst.stat().st_size})
    # Cross-file acceptance: input relationships, not just individual hashes.
    board = json.loads((out/'inputs/storyboard.json').read_text())
    mix = json.loads((out/'inputs/audio/music_mix_report.json').read_text())
    if abs(board['total_duration'] - mix['program_mix_duration_seconds']) > .025:
        raise RuntimeError('Storyboard and mix belong to different timing revisions.')
    for record in records:
        if sha256(source / record['path']) != record['sha256']:
            raise RuntimeError('Source changed during snapshot; retry into a fresh output folder.')
    manifest = {'source_task':source.name, 'snapshotted_at':datetime.now(timezone.utc).isoformat(), 'files':records}
    (out/'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'evidence').mkdir(exist_ok=True)
    print(f'Copied and verified {len(records)} inputs.')


if __name__ == '__main__':
    main()
