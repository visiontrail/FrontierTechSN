from fastapi import APIRouter, HTTPException
import httpx
from backend import database
from backend.models import (
    ProviderCreate,
    ProviderUpdate,
    ProviderResponse,
    ProviderListResponse,
    ProviderCatalogResponse,
    ProviderTestRequest,
    ProviderTestResponse,
)
from backend.provider_catalog import describe_provider_catalog
from backend.pipeline import digester

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("", response_model=ProviderListResponse)
async def get_providers():
    return ProviderListResponse(providers=await database.list_providers())


@router.get("/catalog", response_model=ProviderCatalogResponse)
async def get_provider_catalog():
    return ProviderCatalogResponse(providers=describe_provider_catalog())


@router.post("", response_model=ProviderResponse)
async def add_provider(body: ProviderCreate):
    return await database.create_provider(
        provider_type=body.provider_type,
        name=body.name,
        endpoint=body.endpoint,
        api_key=body.api_key,
        model=body.model,
        is_default=body.is_default,
    )


@router.post("/test", response_model=ProviderTestResponse)
async def test_provider(body: ProviderTestRequest):
    """Send a minimal request to verify the configured model is reachable.

    Accepts an existing provider's id (uses its stored config/key) and/or
    explicit endpoint/model/api_key overrides — useful for testing the form
    before saving. When editing a saved provider with a blank api_key field,
    pass provider_id so the stored key is reused."""
    endpoint = body.endpoint
    model = body.model
    api_key = body.api_key

    if body.provider_id is not None:
        row = await database.get_provider_raw(body.provider_id)
        if row is not None:
            endpoint = endpoint or row["endpoint"]
            model = model or row["model"]
            if not api_key:
                api_key = row["api_key"] or ""

    if not endpoint or not model:
        raise HTTPException(status_code=400, detail="endpoint and model are required")

    try:
        latency_ms = await digester.test_connection(endpoint, model, api_key or "")
        return ProviderTestResponse(ok=True, message=f"Connection OK — {model}", latency_ms=latency_ms)
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        return ProviderTestResponse(ok=False, message=f"HTTP {e.response.status_code}: {detail}")
    except (KeyError, IndexError, TypeError):
        return ProviderTestResponse(ok=False, message="Unexpected response shape (not OpenAI-compatible)")
    except Exception as e:
        return ProviderTestResponse(ok=False, message=str(e) or e.__class__.__name__)


@router.put("/{provider_id}", response_model=ProviderResponse)
async def edit_provider(provider_id: int, body: ProviderUpdate):
    updates = body.model_dump(exclude_unset=True)
    # The edit form never receives the stored secret. A blank field means
    # "leave unchanged", matching the UI copy and the pre-save test behaviour.
    if not updates.get("api_key"):
        updates.pop("api_key", None)
    provider = await database.update_provider(provider_id, **updates)
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.delete("/{provider_id}")
async def remove_provider(provider_id: int):
    existing = await database.get_provider(provider_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    await database.delete_provider(provider_id)
    return {"status": "deleted"}
