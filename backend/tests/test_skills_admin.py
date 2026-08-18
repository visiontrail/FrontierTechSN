import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import skills_admin
from backend.routers.skills import router as skills_router


SKILL_MD = b"""---
name: demo-skill
description: A demo skill used by the upload tests.
---

# Demo

Follow these instructions.
"""


def make_zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


class SkillImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.skills_dir = root / ".claude" / "skills"
        self.state_path = root / "data" / "admin_state.json"
        self.patches = [
            patch.object(skills_admin, "SKILLS_DIR", self.skills_dir),
            patch.object(skills_admin, "_STATE_PATH", self.state_path),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_imports_standalone_skill_document(self):
        skill = skills_admin.import_skill("SKILL.md", SKILL_MD)

        self.assertEqual(skill.name, "demo-skill")
        self.assertTrue((self.skills_dir / "demo-skill" / "SKILL.md").is_file())

    def test_upload_endpoint_loads_skill(self):
        app = FastAPI()
        app.include_router(skills_router)
        response = TestClient(app).post(
            "/api/skills/upload",
            files={"file": ("demo.skill", SKILL_MD, "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["name"], "demo-skill")
        self.assertTrue(response.json()["enabled"])

    def test_admin_enabled_state_drives_sdk_runtime_lists(self):
        skills_admin.import_skill("SKILL.md", SKILL_MD)

        self.assertEqual(skills_admin.runtime_skill_names(), (["demo-skill"], []))

        self.assertTrue(skills_admin.set_enabled("demo-skill", False))
        self.assertEqual(skills_admin.runtime_skill_names(), ([], ["demo-skill"]))

        self.assertTrue(skills_admin.set_enabled("demo-skill", True))
        self.assertEqual(skills_admin.runtime_skill_names(), (["demo-skill"], []))

    def test_imports_wrapped_zip_with_assets(self):
        bundle = make_zip(
            {
                "demo-skill/SKILL.md": SKILL_MD,
                "demo-skill/references/example.txt": b"reference",
            }
        )

        skill = skills_admin.import_skill("demo.skill", bundle)

        self.assertEqual(skill.name, "demo-skill")
        self.assertEqual(
            (self.skills_dir / "demo-skill" / "references" / "example.txt").read_text(),
            "reference",
        )

    def test_rejects_zip_path_traversal(self):
        bundle = make_zip(
            {
                "demo-skill/SKILL.md": SKILL_MD,
                "../outside.txt": b"unsafe",
            }
        )

        with self.assertRaisesRegex(skills_admin.SkillImportError, "traversal"):
            skills_admin.import_skill("demo.zip", bundle)

    def test_rejects_an_existing_skill(self):
        skills_admin.import_skill("SKILL.md", SKILL_MD)

        with self.assertRaises(skills_admin.SkillAlreadyExistsError):
            skills_admin.import_skill("SKILL.md", SKILL_MD)

    def test_rejects_missing_frontmatter(self):
        with self.assertRaisesRegex(skills_admin.SkillImportError, "frontmatter"):
            skills_admin.import_skill("broken.skill", b"# No metadata")


if __name__ == "__main__":
    unittest.main()
