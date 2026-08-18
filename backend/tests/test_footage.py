import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import config
from backend.pipeline import footage


def wikimedia_page(*, license_name="CC BY-SA 4.0", license_code="cc-by-sa-4.0", size=2_000_000):
    return {
        "title": "File:City cyclists.webm",
        "imageinfo": [
            {
                "mime": "video/webm",
                "width": 1280,
                "height": 720,
                "duration": 12.5,
                "size": size,
                "url": "https://upload.wikimedia.org/city.webm",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:City_cyclists.webm",
                "extmetadata": {
                    "LicenseShortName": {"value": license_name},
                    "License": {"value": license_code},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                    "Artist": {"value": '<a href="/wiki/User:Scout">Open Scout</a>'},
                    "ImageDescription": {"value": "<p>Cyclists cross a city street.</p>"},
                    "AttributionRequired": {"value": "true"},
                },
            }
        ],
    }


class FootagePlanTests(unittest.TestCase):
    def test_plan_parser_accepts_fenced_json_and_deduplicates_queries(self):
        raw = """```json
{"queries":[
  {"query":"city cyclists commuting!", "purpose":"Urban movement"},
  {"query":"city cyclists commuting", "purpose":"duplicate"},
  {"query":"solar panels rooftop", "purpose":"Clean energy"}
]}
```"""
        parsed = footage._parse_plan(raw, 4)

        self.assertEqual([item["query"] for item in parsed], [
            "city cyclists commuting",
            "solar panels rooftop",
        ])

    def test_license_gate_allows_only_explicit_open_licenses(self):
        self.assertTrue(footage._is_open_license("Public domain"))
        self.assertTrue(footage._is_open_license("CC BY-SA 4.0"))
        self.assertFalse(footage._is_open_license(""))
        self.assertFalse(footage._is_open_license("CC BY-NC 4.0"))
        self.assertFalse(footage._is_open_license("All rights reserved"))


class WikimediaCandidateTests(unittest.TestCase):
    def test_candidate_retains_provenance_and_cleans_creator_markup(self):
        candidate = footage._candidate_from_page(wikimedia_page(), "landscape")

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["creator"], "Open Scout")
        self.assertEqual(candidate["license"], "CC BY-SA 4.0")
        self.assertTrue(candidate["attribution_required"])
        self.assertEqual(candidate["width"], 1280)

    def test_candidate_rejects_unknown_license_and_large_file(self):
        self.assertIsNone(
            footage._candidate_from_page(
                wikimedia_page(license_name="Copyrighted", license_code="copyright"),
                "landscape",
            )
        )
        with patch.object(config, "FOOTAGE_MAX_BYTES", 1_000):
            self.assertIsNone(
                footage._candidate_from_page(wikimedia_page(size=2_000), "landscape")
            )


class AcquireFootageTests(unittest.IsolatedAsyncioTestCase):
    async def test_acquisition_writes_download_and_auditable_manifest(self):
        candidate = footage._candidate_from_page(wikimedia_page(), "landscape")

        async def fake_search(client, *, query, orientation, limit=16):
            self.assertEqual(query, "city cyclists")
            self.assertEqual(orientation, "landscape")
            return [candidate]

        async def fake_download(client, *, candidate, destination):
            destination.write_bytes(b"public-video")
            return len(b"public-video"), "demo-sha256"

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            script_path = task_dir / "script.txt"
            script_path.write_text("A story about cycling through a changing city.")

            with (
                patch.object(footage, "search_wikimedia", fake_search),
                patch.object(footage, "_download_candidate", fake_download),
            ):
                manifest = await footage.acquire_public_footage(
                    task_id="task-demo",
                    task_dir=task_dir,
                    title="City mobility",
                    script_path=script_path,
                    clip_count=1,
                    orientation="landscape",
                    license_policy="open_only",
                    provider_id=None,
                    ai_endpoint=None,
                    ai_model=None,
                    supplied_queries=["city cyclists"],
                )

            saved = json.loads(
                (task_dir / "footage" / "manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(saved["planner"], "user")
            self.assertEqual(saved["clips"][0]["license"], "CC BY-SA 4.0")
            self.assertEqual(saved["clips"][0]["sha256"], "demo-sha256")
            self.assertTrue(
                (task_dir / saved["clips"][0]["local_path"]).is_file()
            )


if __name__ == "__main__":
    unittest.main()
