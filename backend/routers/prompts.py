from fastapi import APIRouter, HTTPException

from backend import prompts_registry
from backend.models import (
    PromptDetail,
    PromptListResponse,
    PromptUpdate,
)

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("", response_model=PromptListResponse)
async def list_prompts():
    return PromptListResponse(
        prompts=[prompts_registry.to_dict(spec) for spec in prompts_registry.PROMPT_REGISTRY]
    )


@router.get("/{key}", response_model=PromptDetail)
async def get_prompt(key: str):
    spec = prompts_registry.get_spec(key)
    if spec is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompts_registry.to_dict(spec)


@router.put("/{key}", response_model=PromptDetail)
async def update_prompt(key: str, body: PromptUpdate):
    spec = prompts_registry.get_spec(key)
    if spec is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    prompts_registry.write_content(spec, body.content)
    return prompts_registry.to_dict(spec)


@router.post("/{key}/reset", response_model=PromptDetail)
async def reset_prompt(key: str):
    spec = prompts_registry.get_spec(key)
    if spec is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    try:
        prompts_registry.reset_content(spec)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="No default snapshot available for this prompt")
    return prompts_registry.to_dict(spec)
