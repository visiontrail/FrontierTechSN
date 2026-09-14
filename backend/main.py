from backend.crash_logging import configure_crash_logging

# Install before importing pipeline/native dependencies, including direct
# uvicorn launches that bypass scripts/start.sh.
configure_crash_logging()

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.database import init_db, reset_orphaned_account_runs, reset_orphaned_tasks
from backend.logging_setup import configure_logging
from backend.worker import start_worker, stop_worker
from backend.daily_news.scheduler import start_daily_scheduler, stop_daily_scheduler
from backend.routers import (
    daily_news,
    prompts,
    providers,
    settings,
    skills,
    tasks,
    voices,
)
from backend import config
from backend import prompts_registry
from backend.runtime import RUNTIME_IDENTITY
from backend.pipeline.voice_previews import preload_remote_voice_previews

# Not logging.basicConfig: log writes must never block the event loop that is
# streaming subprocess output. See backend/logging_setup.py.
configure_logging(level=logging.INFO)


class _AccessLogErrorsOnly(logging.Filter):
    """Drop uvicorn access-log lines for successful (<400) responses; keep errors.

    Avoids the noise of frequent polling endpoints (e.g. GET /api/providers)
    while still surfacing 4xx/5xx responses.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if args and len(args) >= 5:
            status = args[4]
            if isinstance(status, int) and status < 400:
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(_AccessLogErrorsOnly())

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    prompts_registry.snapshot_defaults()
    await init_db()
    reset = await reset_orphaned_tasks()
    if reset:
        logging.getLogger("backend.main").warning(
            "Reset %d task(s) left in-progress by a previous run to FAILED", reset
        )
    account_reset = await reset_orphaned_account_runs()
    if account_reset:
        logging.getLogger("backend.main").warning(
            "Reset %d interrupted account-operation run(s) to FAILED", account_reset
        )
    start_worker()
    start_daily_scheduler()
    preview_preload_task = asyncio.create_task(
        preload_remote_voice_previews(), name="voice-preview-preload"
    )
    try:
        yield
    finally:
        preview_preload_task.cancel()
        try:
            await preview_preload_task
        except asyncio.CancelledError:
            pass
        await stop_daily_scheduler()
        await stop_worker()


app = FastAPI(title="FrontierTechSN", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)
app.include_router(settings.router)
app.include_router(providers.router)
app.include_router(prompts.router)
app.include_router(skills.router)
app.include_router(voices.router)
app.include_router(daily_news.router)

# Bound to the directory as it stands at startup — this is why OUTPUTS_DIR is
# flagged "restart required" in the Admin console.
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/runtime")
async def runtime_identity():
    return RUNTIME_IDENTITY


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_frontend(full_path: str):
    if full_path.startswith(("api/", "outputs/")):
        raise HTTPException(status_code=404, detail="Not found")

    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run `cd frontend && npm run build` or open http://localhost:5173.",
        )

    requested_path = (FRONTEND_DIST / full_path).resolve()
    if full_path and FRONTEND_DIST.resolve() in requested_path.parents and requested_path.is_file():
        return FileResponse(requested_path)

    return FileResponse(index_path)
