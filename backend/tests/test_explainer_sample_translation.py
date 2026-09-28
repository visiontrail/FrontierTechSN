import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import translate_explainer_samples as migration


def save_project(outputs: Path, project_id: str, **changes) -> tuple[Path, dict, bytes]:
    project = {
        "id": project_id,
        "title": migration.TITLES.get(project_id, ("Existing title",))[0],
        "research": "Existing research",
        "revision": 7,
        "status": "completed",
        "created_at": "2026-09-24T02:00:00+00:00",
        "updated_at": "2026-09-24T03:00:00+00:00",
        "scenes": [{"narration": "Keep this narration.", "visual_prompt": "Keep this visual."}],
        "progress": {"completed": 1, "total": 1, "message": "Finished"},
        "artifacts": {"video_path": "final.mp4", "audio_path": "narration.wav", "duration_seconds": 10.2},
        "logs": ["A retained log message"],
        **changes,
    }
    path = outputs / "explainers" / project_id / "project.json"
    path.parent.mkdir(parents=True)
    original = json.dumps(project, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(original)
    return path, project, original


def test_translation_preserves_delivery_and_backs_up_exact_bytes(tmp_path):
    paths = [save_project(tmp_path, project_id) for project_id in migration.TITLES]
    media = paths[0][0].parent / "final.mp4"
    media.write_bytes(b"original media")

    preview = migration.migrate(tmp_path)
    assert len(preview["changes"]) == 3
    assert preview["backup_dir"] is None
    assert not (tmp_path / "explainer-validation").exists()
    assert all(path.read_bytes() == original for path, _, original in paths)

    report = migration.migrate(tmp_path, apply=True)
    for path, project, original in paths:
        expected = {**project, "title": migration.TITLES[project["id"]][1], "revision": 8}
        assert json.loads(path.read_bytes()) == expected
        assert (Path(report["backup_dir"]) / f"{project['id']}.json").read_bytes() == original
    assert media.read_bytes() == b"original media"
    saved = [path.read_bytes() for path, _, _ in paths]
    assert migration.migrate(tmp_path, apply=True)["changes"] == []
    assert [path.read_bytes() for path, _, _ in paths] == saved
    assert len(list((tmp_path / "explainer-validation").iterdir())) == 1


def test_only_exact_research_line_and_provider_label_are_replaced(tmp_path):
    old, new = migration.TITLES[migration.RESEARCH_SAMPLE]
    research = f"Notes\r\n{old}\r\nKeep {old} inside a sentence.\nCustom 中文 research.\n"
    research_path, _, _ = save_project(tmp_path, migration.RESEARCH_SAMPLE, research=research)
    log_path, _, _ = save_project(tmp_path, migration.LOG_SAMPLE,
                                  logs=["2026-09-24 Provider: DeepSeek 深度求索", "Model: DeepSeek 深度求索", "Retain 中文 note"])
    migration.migrate(tmp_path, apply=True)
    assert json.loads(research_path.read_bytes())["research"] == research.replace(f"\r\n{old}\r\n", f"\r\n{new}\r\n")
    logs = json.loads(log_path.read_bytes())
    assert logs["logs"] == ["2026-09-24 Provider: DeepSeek", "Model: DeepSeek", "Retain 中文 note"]
    assert logs["revision"] == 8


def test_unknown_project_and_customized_sample_are_untouched(tmp_path):
    sample_id = next(iter(migration.TITLES))
    paths = [
        save_project(tmp_path, "exp-ffffffffffffffffffffffff", title=migration.TITLES[sample_id][0]),
        save_project(tmp_path, sample_id, title="My customized 中文 project"),
        save_project(tmp_path, migration.RESEARCH_SAMPLE, title="Custom title", research=migration.TITLES[migration.RESEARCH_SAMPLE][0]),
    ]
    assert migration.migrate(tmp_path, apply=True)["changes"] == []
    assert all(path.read_bytes() == original for path, _, original in paths)


@pytest.mark.parametrize("status", sorted(migration.ACTIVE))
def test_active_project_aborts_before_any_mutation(tmp_path, status):
    project_ids = list(migration.TITLES)
    paths = [save_project(tmp_path, project_ids[0]), save_project(tmp_path, project_ids[1], status=status)]
    with pytest.raises(ValueError, match="generation is active"):
        migration.migrate(tmp_path, apply=True)
    assert all(path.read_bytes() == original for path, _, original in paths)
    assert not (tmp_path / "explainer-validation").exists()


def test_concurrent_edit_is_preserved_and_backup_retains_original(tmp_path):
    path, project, original = save_project(tmp_path, next(iter(migration.TITLES)))
    concurrent = json.dumps({**project, "revision": 8, "title": "A concurrent edit"}).encode("utf-8")
    real_replace = migration._replace_unchanged

    def concurrent_replace(target, expected, replacement):
        target.write_bytes(concurrent)
        real_replace(target, expected, replacement)

    with patch.object(migration, "_replace_unchanged", side_effect=concurrent_replace):
        with pytest.raises(ValueError, match="changed concurrently"):
            migration.migrate(tmp_path, apply=True)
    assert path.read_bytes() == concurrent
    backups = list((tmp_path / "explainer-validation").glob("*/*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert not list(path.parent.glob(".translation-*"))
