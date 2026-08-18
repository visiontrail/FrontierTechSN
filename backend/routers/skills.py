from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend import skills_admin
from backend.models import SkillDetail, SkillListResponse, SkillUpdate

router = APIRouter(prefix="/api/skills", tags=["skills"])


@router.get("", response_model=SkillListResponse)
async def list_skills():
    return SkillListResponse(
        skills=[skills_admin.to_dict(s) for s in skills_admin.list_skills()]
    )


@router.post("/upload", response_model=SkillDetail, status_code=status.HTTP_201_CREATED)
async def upload_skill(file: UploadFile = File(...)):
    filename = file.filename or ""
    try:
        data = await file.read(skills_admin.MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    try:
        skill = await run_in_threadpool(skills_admin.import_skill, filename, data)
    except skills_admin.SkillAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except skills_admin.SkillImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return skills_admin.to_dict(skill, include_body=True)


@router.get("/{name}", response_model=SkillDetail)
async def get_skill(name: str):
    skill = skills_admin.get_skill(name, include_body=True)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skills_admin.to_dict(skill, include_body=True)


@router.put("/{name}", response_model=SkillDetail)
async def update_skill(name: str, body: SkillUpdate):
    if body.enabled is not None:
        if not skills_admin.set_enabled(name, body.enabled):
            raise HTTPException(status_code=404, detail="Skill not found")
    if body.body is not None:
        if not skills_admin.update_body(name, body.body):
            raise HTTPException(status_code=404, detail="Skill not found")
    skill = skills_admin.get_skill(name, include_body=True)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skills_admin.to_dict(skill, include_body=True)
