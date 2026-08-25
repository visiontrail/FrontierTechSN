import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite

from backend import config, database
from backend.models import TaskConfig, TaskStatus


class TaskDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_provider_table_gains_catalog_type_without_changing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "legacy-providers.db"
            connection = await aiosqlite.connect(db_path)
            connection.row_factory = aiosqlite.Row
            await connection.execute(
                """CREATE TABLE providers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    api_key TEXT,
                    model TEXT NOT NULL,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )
            await connection.execute(
                """INSERT INTO providers
                   (name, endpoint, api_key, model, is_default, created_at)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (
                    "Internal gateway",
                    "http://oneapi.yhroot.com/v1/chat/completions",
                    "secret-key",
                    "yinhe-chat",
                    "2026-08-25T00:00:00+00:00",
                ),
            )
            await connection.commit()

            await database._migrate_providers(connection)
            row = (
                await connection.execute_fetchall(
                    "SELECT provider_type, endpoint, api_key, model FROM providers"
                )
            )[0]
            await connection.close()

        self.assertEqual(row["provider_type"], "yinhe")
        self.assertEqual(row["endpoint"], "http://oneapi.yhroot.com/v1/chat/completions")
        self.assertEqual(row["api_key"], "secret-key")
        self.assertEqual(row["model"], "yinhe-chat")

    async def test_provider_catalog_type_round_trips_through_database_response(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "providers.db"
            with (
                patch.object(config, "DB_PATH", db_path),
                patch.object(config, "AI_ENDPOINT", ""),
            ):
                await database.init_db()
                created = await database.create_provider(
                    provider_type="moonshot",
                    name="Kimi production",
                    endpoint="https://api.moonshot.cn/anthropic",
                    api_key="secret-key",
                    model="kimi-k3",
                    is_default=True,
                )
                listed = await database.list_providers()

        self.assertEqual(created.provider_type, "moonshot")
        self.assertEqual(created.api_key_masked, "secr...-key")
        self.assertEqual([provider.provider_type for provider in listed], ["moonshot"])

    async def test_task_status_compare_and_set_has_only_one_concurrent_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tasks.db"
            with patch.object(config, "DB_PATH", db_path):
                await database.init_db()
                task = await database.create_task(
                    "youtube",
                    "https://example.com/video",
                    TaskConfig(),
                )
                results = await asyncio.gather(
                    database.compare_and_set_task_status(
                        task.id,
                        TaskStatus.QUEUED,
                        TaskStatus.COMPOSING,
                        expected_updated_at=task.updated_at,
                    ),
                    database.compare_and_set_task_status(
                        task.id,
                        TaskStatus.QUEUED,
                        TaskStatus.COMPOSING,
                        expected_updated_at=task.updated_at,
                    ),
                )
                refreshed = await database.get_task(task.id)

            self.assertEqual(sorted(results), [False, True])
            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed.status, TaskStatus.COMPOSING)

    async def test_complete_rework_claim_sets_and_conditionally_consumes_publish_suppression(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tasks.db"
            with patch.object(config, "DB_PATH", db_path):
                await database.init_db()
                task = await database.create_task(
                    "youtube",
                    "https://example.com/video",
                    TaskConfig(),
                )
                await database.update_task(task.id, status=TaskStatus.COMPLETE.value)
                complete = await database.get_task(task.id)

                claimed = await database.compare_and_set_task_status(
                    task.id,
                    TaskStatus.COMPLETE,
                    TaskStatus.COMPOSING,
                    expected_updated_at=complete.updated_at,
                    suppress_next_auto_publish=True,
                )
                composing = await database.get_task(task.id)
                await database.update_task(task.id, status=TaskStatus.COMPLETE.value)
                rendered = await database.get_task(task.id)
                consumed = await database.compare_and_set_task_status(
                    task.id,
                    TaskStatus.COMPLETE,
                    TaskStatus.COMPLETE,
                    expected_updated_at=rendered.updated_at,
                    suppress_next_auto_publish=False,
                )
                final = await database.get_task(task.id)

            self.assertTrue(claimed)
            self.assertTrue(composing.suppress_next_auto_publish)
            self.assertTrue(consumed)
            self.assertFalse(final.suppress_next_auto_publish)

    async def test_stale_completion_cannot_clear_newer_publish_suppression(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tasks.db"
            with patch.object(config, "DB_PATH", db_path):
                await database.init_db()
                task = await database.create_task(
                    "youtube",
                    "https://example.com/video",
                    TaskConfig(),
                )
                await database.update_task(
                    task.id,
                    status=TaskStatus.COMPLETE.value,
                    suppress_next_auto_publish=True,
                )
                old_complete = await database.get_task(task.id)
                await database.compare_and_set_task_status(
                    task.id,
                    TaskStatus.COMPLETE,
                    TaskStatus.COMPOSING,
                    expected_updated_at=old_complete.updated_at,
                    suppress_next_auto_publish=True,
                )
                cleared = await database.compare_and_set_task_status(
                    task.id,
                    TaskStatus.COMPLETE,
                    TaskStatus.COMPLETE,
                    expected_updated_at=old_complete.updated_at,
                    suppress_next_auto_publish=False,
                )
                current = await database.get_task(task.id)

            self.assertFalse(cleared)
            self.assertEqual(current.status, TaskStatus.COMPOSING)
            self.assertTrue(current.suppress_next_auto_publish)

    async def test_orphan_reset_preserves_prepublication_rework_suppression(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tasks.db"
            with patch.object(config, "DB_PATH", db_path):
                await database.init_db()
                task = await database.create_task(
                    "youtube",
                    "https://example.com/video",
                    TaskConfig(),
                )
                await database.update_task(
                    task.id,
                    status=TaskStatus.COMPOSING.value,
                    suppress_next_auto_publish=True,
                )
                reset_count = await database.reset_orphaned_tasks()
                refreshed = await database.get_task(task.id)

            self.assertEqual(reset_count, 1)
            self.assertEqual(refreshed.status, TaskStatus.FAILED)
            self.assertTrue(refreshed.suppress_next_auto_publish)
            self.assertFalse(refreshed.publication_safety_hold)

    async def test_orphaned_publication_fails_closed_against_duplicate_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tasks.db"
            with patch.object(config, "DB_PATH", db_path):
                await database.init_db()
                task = await database.create_task(
                    "youtube",
                    "https://example.com/video",
                    TaskConfig(),
                )
                await database.update_task(
                    task.id,
                    status=TaskStatus.PUBLISHING.value,
                    suppress_next_auto_publish=False,
                )
                reset_count = await database.reset_orphaned_tasks()
                refreshed = await database.get_task(task.id)

            self.assertEqual(reset_count, 1)
            self.assertEqual(refreshed.status, TaskStatus.FAILED)
            self.assertFalse(refreshed.suppress_next_auto_publish)
            self.assertTrue(refreshed.publication_safety_hold)
            self.assertIn("automatic republish is suppressed", refreshed.error_message)

    async def test_generated_title_round_trips_through_task_response(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tasks.db"
            with patch.object(config, "DB_PATH", db_path):
                await database.init_db()
                task = await database.create_task(
                    "youtube",
                    "https://example.com/video",
                    TaskConfig(),
                )
                self.assertIsNone(task.generated_title)

                await database.update_task(task.id, generated_title="A New Publication Title")
                refreshed = await database.get_task(task.id)

            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed.generated_title, "A New Publication Title")

    async def test_legacy_task_table_migration_adds_task_runtime_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "legacy.db"
            connection = await aiosqlite.connect(db_path)
            connection.row_factory = aiosqlite.Row
            await connection.execute(
                """CREATE TABLE tasks (
                    id TEXT PRIMARY KEY,
                    scheduled_at TEXT,
                    thumbnail_path TEXT
                )"""
            )
            await connection.commit()

            await database._migrate_tasks(connection)
            columns = {
                row["name"]
                for row in await connection.execute_fetchall("PRAGMA table_info(tasks)")
            }
            await connection.close()

            self.assertIn("generated_title", columns)
            self.assertIn("suppress_next_auto_publish", columns)
            self.assertIn("publication_safety_hold", columns)

    async def test_legacy_failed_publication_suppression_migrates_to_safety_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "legacy-hold.db"
            connection = await aiosqlite.connect(db_path)
            connection.row_factory = aiosqlite.Row
            await connection.execute(
                """CREATE TABLE tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    suppress_next_auto_publish INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )
            await connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?)",
                ("legacy-hold", TaskStatus.FAILED.value, 1, "2026-08-20T00:00:00+00:00"),
            )
            await connection.commit()

            await database._migrate_tasks(connection)
            row = (
                await connection.execute_fetchall(
                    "SELECT suppress_next_auto_publish, publication_safety_hold "
                    "FROM tasks WHERE id = ?",
                    ("legacy-hold",),
                )
            )[0]
            await connection.close()

            self.assertEqual(row["suppress_next_auto_publish"], 0)
            self.assertEqual(row["publication_safety_hold"], 1)

    async def test_reset_orphaned_tasks_closes_connection_after_lock_error(self):
        connection = AsyncMock()
        connection.execute.side_effect = sqlite3.OperationalError("database is locked")

        with patch.object(database, "get_db", AsyncMock(return_value=connection)):
            with self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
                await database.reset_orphaned_tasks()

        connection.rollback.assert_awaited_once()
        connection.close.assert_awaited_once()

    async def test_claim_account_run_closes_transaction_when_cancelled(self):
        connection = AsyncMock()
        connection.execute.side_effect = asyncio.CancelledError

        with patch.object(database, "get_db", AsyncMock(return_value=connection)):
            with self.assertRaises(asyncio.CancelledError):
                await database.claim_next_account_run()

        connection.rollback.assert_awaited_once()
        connection.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
