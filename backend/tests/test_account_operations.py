import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from backend import config, database
from backend.account_ops.opencode import OpenCodeError, _assistant_text
from backend.account_ops.orchestrator import (
    _content_object,
    _ensure_account,
    _prepare_publish_image,
    _publish,
    _render_prompt,
    _verify_account,
)
from backend.account_ops.schedule import next_daily_run, next_scheduled_run
from backend.account_ops.worker import get_worker_status
from backend.account_ops.x_engagement import (
    OperationalAgentResult,
    parse_engagement_result,
    recover_action_urls,
    validate_engagement_result,
)
from backend.models import AccountAutomationFeature, AccountRunStatus
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult


class AccountScheduleTests(unittest.TestCase):
    def test_daily_wall_clock_is_converted_across_timezones(self):
        after = datetime(2026, 8, 4, 0, 30, tzinfo=timezone.utc)

        self.assertEqual(
            next_daily_run("09:00", "Asia/Singapore", after=after),
            "2026-08-04T01:00:00+00:00",
        )

    def test_passed_wall_clock_moves_to_the_next_day(self):
        after = datetime(2026, 8, 4, 2, 0, tzinfo=timezone.utc)

        self.assertEqual(
            next_daily_run("09:00", "Asia/Singapore", after=after),
            "2026-08-05T01:00:00+00:00",
        )

    def test_multiple_daily_times_choose_the_next_local_occurrence(self):
        after = datetime(2026, 8, 4, 2, 0, tzinfo=timezone.utc)

        self.assertEqual(
            next_scheduled_run(
                ["01:00", "11:00", "23:00"],
                "Asia/Singapore",
                after=after,
            ),
            "2026-08-04T03:00:00+00:00",
        )


class AccountDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addAsyncCleanup(self._cleanup)
        self.db_path = Path(self.temp.name) / "account-operations.db"
        self.db_patch = patch.object(config, "DB_PATH", self.db_path)
        self.db_patch.start()
        await database.init_db()

    async def _cleanup(self):
        self.db_patch.stop()
        self.temp.cleanup()

    async def test_init_seeds_the_quiet_atlas_commission(self):
        automations = await database.list_account_automations()

        self.assertEqual(len(automations), 2)
        by_feature = {automation.feature_type: automation for automation in automations}
        history = by_feature[AccountAutomationFeature.TODAY_IN_HISTORY]
        engagement = by_feature[AccountAutomationFeature.X_ENGAGEMENT]
        self.assertEqual(history.account_handle, "AQuietAtlas")
        self.assertEqual(engagement.account_handle, "AQuietAtlas")
        self.assertEqual(engagement.schedule_times, ["01:00", "04:30", "23:00"])
        self.assertTrue(history.enabled)
        self.assertIsNotNone(engagement.next_run_at)

    async def test_multiple_schedule_update_recomputes_the_next_run(self):
        automation = next(
            item
            for item in await database.list_account_automations()
            if item.feature_type == AccountAutomationFeature.X_ENGAGEMENT
        )

        updated = await database.update_account_automation(
            automation.id,
            {"schedule_times": ["02:15", "18:45"]},
        )

        self.assertEqual(updated.schedule_times, ["02:15", "18:45"])
        self.assertEqual(updated.schedule_time, "02:15")
        self.assertNotEqual(updated.next_run_at, automation.next_run_at)

    async def test_prompt_edit_preserves_the_existing_schedule(self):
        automation = (await database.list_account_automations())[0]

        updated = await database.update_account_automation(
            automation.id,
            {"prompt_template": automation.prompt_template + "\nPrefer primary sources."},
        )

        self.assertEqual(updated.next_run_at, automation.next_run_at)

    async def test_manual_run_is_claimed_once_with_an_account_snapshot(self):
        automation = (await database.list_account_automations())[0]
        created = await database.create_account_run(automation, trigger="manual")

        claimed = await database.claim_next_account_run()
        second = await database.claim_next_account_run()

        self.assertEqual(claimed.id, created.id)
        self.assertEqual(claimed.status, AccountRunStatus.PLANNING)
        self.assertEqual(claimed.account_handle, "AQuietAtlas")
        self.assertIsNone(second)

    async def test_due_schedule_is_enqueued_only_once(self):
        automation = (await database.list_account_automations())[0]
        db = await database.get_db()
        await db.execute(
            "UPDATE account_automations SET next_run_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", automation.id),
        )
        await db.commit()
        await db.close()

        first = await database.enqueue_due_account_runs()
        second = await database.enqueue_due_account_runs()
        runs = await database.list_account_runs()

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].trigger, "scheduled")

    async def test_pause_clears_next_run_and_prevents_enqueue(self):
        automation = (await database.list_account_automations())[0]
        self.assertIsNotNone(automation.next_run_at)

        paused = await database.update_account_automation(
            automation.id, {"enabled": False}
        )

        self.assertFalse(paused.enabled)
        self.assertIsNone(paused.next_run_at)

        enqueued = await database.enqueue_due_account_runs()
        self.assertEqual(enqueued, 0)

    async def test_resume_restores_next_run(self):
        automation = (await database.list_account_automations())[0]
        await database.update_account_automation(automation.id, {"enabled": False})

        resumed = await database.update_account_automation(
            automation.id, {"enabled": True}
        )

        self.assertTrue(resumed.enabled)
        self.assertIsNotNone(resumed.next_run_at)


class AccountWorkerStatusTests(unittest.TestCase):
    def test_status_reports_alive_false_when_worker_not_started(self):
        status = get_worker_status()

        self.assertFalse(status["worker_alive"])
        self.assertIn("poll_interval", status)

    def test_status_reports_alive_true_when_task_is_active(self):
        from unittest.mock import MagicMock

        from backend.account_ops import worker

        original_task = worker._worker_task
        try:
            mock_task = MagicMock()
            mock_task.done.return_value = False
            worker._worker_task = mock_task
            status = get_worker_status()
            self.assertTrue(status["worker_alive"])
        finally:
            worker._worker_task = original_task

    def test_status_reports_alive_false_when_task_is_done(self):
        from unittest.mock import MagicMock

        from backend.account_ops import worker

        original_task = worker._worker_task
        try:
            mock_task = MagicMock()
            mock_task.done.return_value = True
            worker._worker_task = mock_task
            status = get_worker_status()
            self.assertFalse(status["worker_alive"])
        finally:
            worker._worker_task = original_task


class AccountOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_variables_render_without_touching_json_language(self):
        rendered = _render_prompt(
            "On {month_name} {day}, return JSON for {date} in {year}.",
            "2026-08-04",
        )

        self.assertEqual(
            rendered,
            "On August 4, return JSON for 2026-08-04 in 2026.",
        )

    def test_content_contract_accepts_a_sourced_post(self):
        content = _content_object(
            {
                "title": "A consequential event",
                "year": 1914,
                "event_summary": "A verified account of what happened.",
                "historical_reflection": "Institutions turn choices into consequences.",
                "post_text": "In 1914, a consequential event altered institutions and public life. History rarely repeats cleanly; it preserves the cost of choices long after their authors are gone.",
                "image_prompt": "A period-authentic documentary scene without text.",
                "source_notes": ["Archive A", "Museum B"],
            }
        )

        self.assertEqual(content["year"], 1914)

    def test_content_contract_compacts_an_oversized_post(self):
        content = _content_object(
            {
                "title": "Event",
                "year": 1900,
                "event_summary": "Summary",
                "historical_reflection": "Reflection",
                "post_text": "A historically grounded sentence " * 12,
                "image_prompt": "Scene",
                "source_notes": ["A", "B"],
            }
        )

        self.assertLessEqual(len(content["post_text"]), 280)
        self.assertTrue(content["post_text"].endswith("…"))

    def test_content_contract_selects_final_content_json_after_tool_events(self):
        planned = {
            "title": "Event",
            "year": 1900,
            "event_summary": "Summary",
            "historical_reflection": "Reflection",
            "post_text": "In 1900, this event changed public life and institutions. Its consequences show how choices can outlive the people who made them.",
            "image_prompt": "Scene",
            "source_notes": ["A", "B"],
        }
        mixed_output = (
            'Tool returned [{"type":"text","text":"working"}]\n'
            "Final answer:\n"
            + json.dumps(planned)
        )

        content = _content_object(mixed_output)

        self.assertEqual(content["title"], "Event")

    def test_publish_image_is_compacted_for_browser_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (1800, 1200), (36, 48, 58)).save(source)

            output = _prepare_publish_image(source, root)

            self.assertEqual(output.suffix, ".jpg")
            self.assertLess(output.stat().st_size, source.stat().st_size)
            with Image.open(output) as image:
                self.assertLessEqual(image.width, 1400)
                self.assertLessEqual(image.height, 1400)

    async def test_publish_guard_rejects_the_wrong_browser_account(self):
        result = OpenCLIResult(
            args=("twitter", "whoami"),
            returncode=0,
            stdout='{"logged_in":true,"username":"SomeoneElse"}',
            stderr="",
        )
        with patch(
            "backend.account_ops.orchestrator.run_opencli",
            AsyncMock(return_value=result),
        ) as opencli:
            with self.assertRaisesRegex(OpenCLIError, "Refusing to publish"):
                await _verify_account("AQuietAtlas")

        self.assertIn("ephemeral", opencli.await_args.args[0])

    async def test_account_guard_prompts_a_switch_then_rechecks(self):
        automation = database.AccountAutomationResponse(
            id="auto_test",
            created_at="2026-08-07T00:00:00+00:00",
            updated_at="2026-08-07T00:00:00+00:00",
            account_handle="AQuietAtlas",
        )
        wrong = OpenCLIResult(
            args=("twitter", "whoami"), returncode=0,
            stdout='{"logged_in":true,"username":"SomeoneElse"}', stderr="",
        )
        right = OpenCLIResult(
            args=("twitter", "whoami"), returncode=0,
            stdout='{"logged_in":true,"username":"AQuietAtlas"}', stderr="",
        )
        switched = OperationalAgentResult(
            text='{"account_handle":"AQuietAtlas","verified":true}',
            session_id="session-1",
            raw="raw",
        )
        with (
            patch(
                "backend.account_ops.orchestrator.run_opencli",
                AsyncMock(side_effect=[wrong, right]),
            ),
            patch(
                "backend.account_ops.orchestrator.run_x_operational_agent",
                AsyncMock(return_value=switched),
            ) as switch_agent,
        ):
            account, audit = await _ensure_account(automation, run_id="acct_test")

        self.assertEqual(account["username"], "AQuietAtlas")
        self.assertEqual(audit["agent_session_id"], "session-1")
        switch_agent.assert_awaited_once()

    async def test_a_publish_that_reached_x_before_dying_is_not_reported_failed(self):
        post_text = "On 5 August 1858 the first transatlantic cable carried its first message."
        timeline = OpenCLIResult(
            args=("twitter", "tweets"),
            returncode=0,
            stdout=(
                '[{"id":"1234","text":"On 5 August 1858 the first transatlantic cable '
                'carried its first message.","url":"https://x.com/AQuietAtlas/status/1234"}]'
            ),
            stderr="",
        )

        async def opencli(args, **kwargs):
            if args[1] == "post":
                raise OpenCLIError("CDP command Runtime.evaluate timed out after 115s")
            return timeline

        with patch("backend.account_ops.orchestrator.run_opencli", opencli):
            url, post_id, raw = await _publish(post_text, Path("image.jpg"), "AQuietAtlas")

        self.assertEqual(url, "https://x.com/AQuietAtlas/status/1234")
        self.assertEqual(post_id, "1234")
        self.assertIn("recovered_from_error", raw)

    async def test_publish_result_from_the_wrong_account_is_rejected(self):
        wrong_account = OpenCLIResult(
            args=("twitter", "post"),
            returncode=0,
            stdout=(
                '[{"status":"success","id":"1234","url":'
                '"https://x.com/SomeoneElse/status/1234"}]'
            ),
            stderr="",
        )

        with patch(
            "backend.account_ops.orchestrator.run_opencli",
            AsyncMock(return_value=wrong_account),
        ) as opencli:
            with self.assertRaisesRegex(OpenCLIError, "wrong account @SomeoneElse"):
                await _publish("A commissioned post.", Path("image.jpg"), "AQuietAtlas")

        self.assertIn("ephemeral", opencli.await_args.args[0])

    async def test_a_publish_absent_from_the_timeline_is_declared_safe_to_retry(self):
        empty = OpenCLIResult(
            args=("twitter", "tweets"), returncode=0, stdout="[]", stderr=""
        )

        async def opencli(args, **kwargs):
            if args[1] == "post":
                raise OpenCLIError("CDP command Runtime.evaluate timed out after 115s")
            return empty

        with patch("backend.account_ops.orchestrator.run_opencli", opencli):
            with self.assertRaisesRegex(OpenCLIError, "safe to retry"):
                await _publish("Nothing went out.", Path("image.jpg"), "AQuietAtlas")

    async def test_an_unreadable_timeline_leaves_the_publication_state_unknown(self):
        async def opencli(args, **kwargs):
            raise OpenCLIError(f"OpenCLI twitter {args[1]} failed with exit 1")

        with patch("backend.account_ops.orchestrator.run_opencli", opencli):
            with self.assertRaisesRegex(OpenCLIError, "UNKNOWN"):
                await _publish("Ambiguous.", Path("image.jpg"), "AQuietAtlas")


class OpenCodeOutputTests(unittest.TestCase):
    def test_json_event_stream_recovers_assistant_text_and_session(self):
        stdout = "\n".join(
            [
                '{"type":"step_start","sessionID":"ses_123","part":{}}',
                '{"type":"text","sessionID":"ses_123","part":{"text":"{\\"ok\\":true}"}}',
            ]
        )

        session_id, text = _assistant_text(stdout)

        self.assertEqual(session_id, "ses_123")
        self.assertEqual(text, '{"ok":true}')

    def test_empty_event_stream_is_an_error(self):
        with self.assertRaises(OpenCodeError):
            _assistant_text('{"type":"step_finish","sessionID":"ses_123"}')


class EngagementResultTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addAsyncCleanup(self.temp.cleanup)
        self.db_patch = patch.object(config, "DB_PATH", Path(self.temp.name) / "ops.db")
        self.db_patch.start()
        self.addAsyncCleanup(self.db_patch.stop)
        await database.init_db()
        self.automation = next(
            item
            for item in await database.list_account_automations()
            if item.feature_type == AccountAutomationFeature.X_ENGAGEMENT
        )

    def result(self) -> dict:
        return {
            "account_handle": "AQuietAtlas",
            "account_switched": False,
            "following_feed_used": True,
            "scanned_posts": 12,
            "replies": [
                {
                    "target_url": "https://x.com/MapArchive/status/100",
                    "target_author": "MapArchive",
                    "reply_text": "The faded rail spur says more than the border line—the map still remembers how people actually moved.",
                    "result_url": "https://x.com/AQuietAtlas/status/200",
                    "has_media": True,
                    "media_explanation": "Grok describes a 1912 railway map with a discontinued branch line.",
                    "is_repost": False,
                    "explained_original_url": None,
                }
            ],
            "quote_reposts": [],
            "skipped": [],
            "notes": [],
        }

    async def test_media_reply_requires_and_preserves_grok_explanation(self):
        validated = validate_engagement_result(self.result(), self.automation)

        self.assertEqual(len(validated["replies"]), 1)
        self.assertIn("railway map", validated["replies"][0]["media_explanation"])

    async def test_browser_element_array_before_final_audit_is_ignored(self):
        verbose_output = (
            "The Grok actions button is [86].\n"
            'Intermediate state: {"status":"working"}\n'
            "All writes are verified. Returning the final audit:\n"
            + json.dumps(self.result())
        )

        parsed = parse_engagement_result(verbose_output)
        recovered = await recover_action_urls(verbose_output, self.automation)

        self.assertEqual(parsed["account_handle"], "AQuietAtlas")
        self.assertEqual(
            recovered["replies"][0]["result_url"],
            "https://x.com/AQuietAtlas/status/200",
        )

    async def test_media_reply_without_grok_explanation_is_rejected(self):
        result = self.result()
        result["replies"][0]["media_explanation"] = None

        with self.assertRaisesRegex(OpenCLIError, "without preserving Grok"):
            validate_engagement_result(result, self.automation)

    async def test_recent_target_is_rejected_to_prevent_duplicate_engagement(self):
        result = self.result()

        with self.assertRaisesRegex(OpenCLIError, "duplicate or excluded"):
            validate_engagement_result(
                result,
                self.automation,
                excluded_urls={"https://x.com/MapArchive/status/100"},
            )

    async def test_result_url_must_belong_to_the_commissioned_account(self):
        result = self.result()
        result["replies"][0]["result_url"] = (
            "https://x.com/SomeoneElse/status/200"
        )

        with self.assertRaisesRegex(OpenCLIError, "belongs to @SomeoneElse"):
            validate_engagement_result(result, self.automation)

    async def test_missing_reply_url_is_recovered_from_the_target_thread(self):
        result = self.result()
        result["replies"][0]["result_url"] = None
        thread = OpenCLIResult(
            args=("twitter", "thread"),
            returncode=0,
            stdout=(
                '[{"author":"MapArchive","text":"Original","url":'
                '"https://x.com/MapArchive/status/100"},'
                '{"author":"AQuietAtlas","text":"@MapArchive The faded rail spur '
                'says more than the border line—the map still remembers how people '
                'actually moved.","url":"https://x.com/AQuietAtlas/status/200"}]'
            ),
            stderr="",
        )

        with patch(
            "backend.account_ops.x_engagement.run_opencli",
            AsyncMock(return_value=thread),
        ) as opencli:
            recovered = await recover_action_urls(result, self.automation)

        self.assertEqual(
            recovered["replies"][0]["result_url"],
            "https://x.com/AQuietAtlas/status/200",
        )
        self.assertIn("ephemeral", opencli.await_args.args[0])

    async def test_wrong_account_result_stops_without_attempting_recovery(self):
        result = self.result()
        result["replies"][0]["result_url"] = (
            "https://x.com/SomeoneElse/status/200"
        )

        with (
            patch(
                "backend.account_ops.x_engagement.run_opencli", AsyncMock()
            ) as opencli,
            self.assertRaisesRegex(OpenCLIError, "wrong X account @SomeoneElse"),
        ):
            await recover_action_urls(result, self.automation)

        opencli.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
