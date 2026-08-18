import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from backend import config, database
from backend.models import (
    ContentPlanItemCreate,
    ContentPlanStatus,
    ContentSeriesCreate,
    PublicationStatus,
    TaskConfig,
    TaskStatus,
)
from backend.pipeline.extractors.topic import extract_topic
from backend.publishing import run_auto_publish_pipeline


class ContentPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            config, "DB_PATH", Path(self.directory.name) / "planning.db"
        )
        self.db_patch.start()
        await database.init_db()

    async def asyncTearDown(self):
        self.db_patch.stop()
        self.directory.cleanup()

    async def test_scheduled_plan_materializes_linked_video_task(self):
        series = await database.create_content_series(
            ContentSeriesCreate(
                name="Borderlands",
                theme="Countries and identity",
                description="How borders shape daily life.",
            )
        )
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                series_id=series.id,
                title="The border that crosses a library",
                brief="Explain the history, the people who use it, and what the border reveals.",
                episode_number=3,
                generation_at="2030-04-10T01:00:00Z",
                publish_at="2030-04-11T09:00:00Z",
                platform="YouTube",
            )
        )

        self.assertEqual(item.status, ContentPlanStatus.SCHEDULED)
        self.assertIsNotNone(item.task_id)
        task = await database.get_task(item.task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.source_type.value, "topic")
        self.assertEqual(task.scheduled_at, "2030-04-10T01:00:00+00:00")
        self.assertEqual(task.planned_publish_at, "2030-04-11T09:00:00+00:00")
        self.assertEqual(task.origin_type, "content_plan")
        self.assertEqual(task.origin_id, item.id)
        self.assertEqual(
            task.origin_label,
            "Borderlands · EP 03 · The border that crosses a library",
        )

    async def test_youtube_plan_preserves_source_and_generation_configuration(self):
        config = TaskConfig(
            target_duration_minutes=20,
            script_format="dialogue",
            speaker_count=2,
            tts_model="vibevoice-1.5b",
            video_orientation="portrait",
            captions_enabled=True,
            footage_enabled=True,
            footage_provider="hybrid",
        )
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="A planned YouTube adaptation",
                brief="Use the linked source while following this editorial angle closely.",
                source_type="youtube",
                source_url="https://www.youtube.com/watch?v=planned-source",
                generation_at="2030-04-10T01:00:00Z",
                task_config=config,
            )
        )

        task = await database.get_task(item.task_id)
        self.assertEqual(item.source_type, "youtube")
        self.assertEqual(item.source_url, "https://www.youtube.com/watch?v=planned-source")
        self.assertEqual(task.source_type.value, "youtube")
        self.assertEqual(task.source_url, item.source_url)
        self.assertEqual(task.config.target_duration_minutes, 20)
        self.assertEqual(task.config.video_orientation, "portrait")
        self.assertTrue(task.config.captions_enabled)

    async def test_youtube_plan_requires_a_source_url(self):
        with self.assertRaisesRegex(ValueError, "YouTube URL"):
            ContentPlanItemCreate(
                title="Missing source",
                brief="This planned video intentionally omits its required source URL.",
                source_type="youtube",
            )

        with self.assertRaisesRegex(ValueError, "valid YouTube URL"):
            ContentPlanItemCreate(
                title="Wrong source",
                brief="This planned video points at a non-YouTube source URL.",
                source_type="youtube",
                source_url="https://example.com/video",
            )

        with self.assertRaisesRegex(ValueError, "identify a YouTube video"):
            ContentPlanItemCreate(
                title="YouTube home page",
                brief="This URL has the right host but does not identify a video.",
                source_type="youtube",
                source_url="https://www.youtube.com/",
            )

    async def test_editing_queued_plan_updates_source_and_configuration(self):
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="Editable queued plan",
                brief="Start as a research topic and then attach the final source later.",
                generation_at="2030-04-10T01:00:00Z",
            )
        )
        updated = await database.update_content_plan_item(
            item.id,
            ContentPlanItemCreate(
                title=item.title,
                brief=item.brief,
                source_type="youtube",
                source_url="https://youtu.be/final-source",
                generation_at=item.generation_at,
                task_config=TaskConfig(target_duration_minutes=15),
            ),
        )

        task = await database.get_task(updated.task_id)
        self.assertEqual(task.source_type.value, "youtube")
        self.assertEqual(task.source_url, "https://youtu.be/final-source")
        self.assertEqual(task.config.target_duration_minutes, 15)

    async def test_rescheduling_planned_task_moves_editorial_clock_atomically(self):
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="A capital built from scratch",
                brief="Compare the political goals and lived results of planned capital cities.",
                generation_at="2030-05-01T01:00:00Z",
                publish_at="2030-05-03T01:00:00Z",
            )
        )

        await database.reschedule_task(item.task_id, "2030-05-02T02:30:00+00:00")

        task = await database.get_task(item.task_id)
        refreshed = await database.get_content_plan_item(item.id)
        self.assertEqual(task.scheduled_at, "2030-05-02T02:30:00+00:00")
        self.assertEqual(refreshed.generation_at, task.scheduled_at)

    async def test_due_plan_is_prioritized_and_future_plan_stays_parked(self):
        manual = await database.create_task(
            "youtube", "https://example.com/manual", config=TaskConfig()
        )
        due_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="A scheduled priority",
                brief="Verify that a due editorial plan runs ahead of an older manual queue item.",
                generation_at=due_at,
                publish_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            )
        )

        selected = await database.get_next_queued_task()
        self.assertEqual(selected.id, item.task_id)
        self.assertNotEqual(selected.id, manual.id)

        await database.update_task(item.task_id, status=TaskStatus.EXTRACTING.value)
        await database.update_task(manual.id, status=TaskStatus.COMPLETE.value)
        future = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="A parked future plan",
                brief="Verify that the worker cannot claim this plan before its release time.",
                generation_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                publish_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            )
        )
        self.assertIsNotNone(future.task_id)
        self.assertIsNone(await database.get_next_queued_task())

    async def test_finished_plan_requires_approval_before_manual_publication(self):
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="The country with two calendars",
                brief="Build a sourced explainer about calendar systems and civic life.",
                generation_at="2030-06-01T01:00:00Z",
                publish_at="2030-06-02T01:00:00Z",
            )
        )
        await database.update_task(item.task_id, status=TaskStatus.COMPLETE.value)

        review = await database.get_content_plan_item(item.id)
        self.assertEqual(review.status, ContentPlanStatus.REVIEW)
        self.assertEqual(review.publication_status, PublicationStatus.AWAITING_REVIEW)
        with self.assertRaisesRegex(ValueError, "approval"):
            await database.record_manual_publication(item.id, "https://example.com/watch")

        approved = await database.approve_content_plan_item(item.id)
        self.assertEqual(approved.status, ContentPlanStatus.READY)
        self.assertEqual(approved.publication_status, PublicationStatus.APPROVED)
        published = await database.record_manual_publication(
            item.id, "https://example.com/watch"
        )
        self.assertEqual(published.status, ContentPlanStatus.PUBLISHED)
        self.assertEqual(published.publication_status, PublicationStatus.PUBLISHED)
        self.assertEqual(published.publication_url, "https://example.com/watch")

    async def test_auto_publication_is_blocked_by_default_and_without_adapter(self):
        item = await database.create_content_plan_item(
            ContentPlanItemCreate(
                title="Why this island changed names",
                brief="Trace the naming history without flattening competing local perspectives.",
                generation_at="2030-07-01T01:00:00Z",
                publish_at="2030-07-02T01:00:00Z",
                platform="YouTube",
                auto_publish_requested=True,
            )
        )
        await database.update_task(item.task_id, status=TaskStatus.COMPLETE.value)
        task = await database.get_task(item.task_id)

        with patch.object(config, "VIDEO_AUTO_PUBLISH_ENABLED", False):
            result = await run_auto_publish_pipeline(task)
        self.assertEqual(result.action, "awaiting_review")
        self.assertIn("disabled", result.reason)

        with patch.object(config, "VIDEO_AUTO_PUBLISH_ENABLED", True):
            result = await run_auto_publish_pipeline(task)
        self.assertEqual(result.action, "awaiting_review")
        self.assertIn("No automatic publisher", result.reason)

    async def test_topic_extractor_preserves_editorial_brief(self):
        content = await extract_topic(
            "The map that created a country",
            "Explain which map mattered, who commissioned it, and how later borders diverged.",
        )

        self.assertEqual(content.source_type, "topic")
        self.assertEqual(content.title, "The map that created a country")
        self.assertTrue(content.metadata["research_required"])
        self.assertIn("later borders", content.text)


if __name__ == "__main__":
    unittest.main()
