from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend import explainers as studio

router = APIRouter(prefix="/api/explainers", tags=["explainers"])


def _call(function, *args):
    try:
        return function(*args)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Explainer project or artifact not found") from exc
    except studio.ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("")
async def list_projects():
    projects = studio.list_projects()
    positions = studio.queue_positions(projects)
    return {"projects": [studio.public_project(p, positions) for p in projects]}


@router.get("/defaults")
async def defaults():
    return _call(studio.defaults)


@router.put("/defaults")
async def save_defaults(data: studio.Defaults):
    return _call(studio.save_defaults, data)


@router.post("", status_code=201)
async def create_project(data: studio.ProjectInput):
    return studio.public_project(_call(studio.create_project, data))


@router.get("/queue")
async def queue_state():
    return await studio.queue_state()


@router.get("/{project_id}")
async def get_project(project_id: str):
    return studio.public_project(_call(studio.read_project, project_id))


@router.put("/{project_id}")
async def update_project(project_id: str, data: studio.ProjectUpdate):
    return studio.public_project(_call(studio.update_project, project_id, data))


@router.post("/{project_id}/script", status_code=202)
async def write_script(project_id: str):
    return studio.public_project(_call(studio.start, project_id, "script"))


@router.post("/{project_id}/render", status_code=202)
async def render_project(project_id: str):
    return studio.public_project(_call(studio.start, project_id, "render"))


@router.post("/{project_id}/queue", status_code=202)
async def queue_project(project_id: str):
    return studio.public_project(_call(studio.start, project_id, "full"))


@router.post("/{project_id}/dequeue")
async def remove_from_queue(project_id: str):
    return studio.public_project(_call(studio.remove_from_queue, project_id))


@router.post("/{project_id}/cancel")
async def cancel(project_id: str):
    _call(studio.read_project, project_id)
    return studio.public_project(await studio.cancel(project_id))


@router.post("/{project_id}/pause")
async def pause(project_id: str):
    _call(studio.read_project, project_id)
    try:
        return studio.public_project(await studio.pause(project_id))
    except studio.ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{project_id}/assets/{relative:path}")
async def asset(project_id: str, relative: str):
    _call(studio.read_project, project_id)
    return FileResponse(_call(studio.asset_path, project_id, relative))
