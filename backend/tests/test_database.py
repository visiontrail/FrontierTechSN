import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite

from backend import config, database
from backend.models import TaskConfig


class TaskDatabaseTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_legacy_task_table_migration_adds_generated_title(self):
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
