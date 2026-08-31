from fastapi import APIRouter, HTTPException
import httpx
import time
from backend import database
from backend.models import (
    ProviderCreate,
    ProviderUpdate,
    ProviderResponse,
    ProviderListResponse,
    ProviderCatalogResponse,
    ProviderTestRequest,
    ProviderTestResponse,
    ProviderKeyTestResult,
    ProviderRouteTestResponse,
)
from backend.provider_credentials import decode_api_keys, key_identifier, redact_api_keys
from backend.provider_catalog import describe_provider_catalog
from backend.pipeline import digester, model_router

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("", response_model=ProviderListResponse)
async def get_providers():
    return ProviderListResponse(providers=await database.list_providers())


@router.get("/catalog", response_model=ProviderCatalogResponse)
async def get_provider_catalog():
    return ProviderCatalogResponse(providers=describe_provider_catalog())


@router.post("", response_model=ProviderResponse)
async def add_provider(body: ProviderCreate):
    try:
        return await database.create_provider(
            provider_type=body.provider_type,
            name=body.name,
            endpoint=body.endpoint,
            api_key=body.api_key,
            api_keys=body.api_keys,
            model=body.model,
            is_default=body.is_default,
            route_role=body.route_role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
    api_keys = list(body.api_keys or [])

    if body.provider_id is not None:
        row = await database.get_provider_raw(body.provider_id)
        if row is not None:
            endpoint = endpoint or row["endpoint"]
            model = model or row["model"]
            if not api_keys and not api_key:
                if str(row["route_role"] or "standalone") == "primary":
                    api_keys = decode_api_keys(row["api_keys_json"])
                api_key = row["api_key"] or ""

    if not endpoint or not model:
        raise HTTPException(status_code=400, detail="endpoint and model are required")

    keys = api_keys or ([api_key] if api_key else [])
    if not keys:
        return ProviderTestResponse(ok=False, message="At least one API key is required")

    results: list[ProviderKeyTestResult] = []
    for key in keys:
        key_id = key_identifier(key)
        try:
            latency_ms = await digester.test_connection(endpoint, model, key)
            results.append(
                ProviderKeyTestResult(
                    key_id=key_id,
                    ok=True,
                    message="Connection OK",
                    latency_ms=latency_ms,
                )
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            detail = detail.replace(key, "<redacted>")
            results.append(
                ProviderKeyTestResult(
                    key_id=key_id,
                    ok=False,
                    message=f"HTTP {exc.response.status_code}: {detail}",
                )
            )
        except (KeyError, IndexError, TypeError):
            results.append(
                ProviderKeyTestResult(
                    key_id=key_id,
                    ok=False,
                    message="Unexpected response shape (not OpenAI-compatible)",
                )
            )
        except Exception as exc:  # noqa: BLE001 - provider diagnostic boundary
            message = (str(exc) or exc.__class__.__name__).replace(key, "<redacted>")
            results.append(
                ProviderKeyTestResult(key_id=key_id, ok=False, message=message)
            )

    succeeded = sum(1 for result in results if result.ok)
    return ProviderTestResponse(
        ok=succeeded == len(results),
        message=f"{succeeded}/{len(results)} API keys passed — {model}",
        latency_ms=max(
            (result.latency_ms or 0 for result in results if result.ok),
            default=None,
        ),
        keys_tested=len(results),
        keys_succeeded=succeeded,
        key_results=results,
    )


@router.post("/route-test", response_model=ProviderRouteTestResponse)
async def test_configured_route():
    """Exercise the same primary-pool and backup route used by real tasks."""
    primary = await database.get_provider_raw(None)
    if primary is None or str(primary["route_role"] or "standalone") != "primary":
        return ProviderRouteTestResponse(ok=False, message="No primary provider configured")
    rows = await database.list_providers_raw()
    secrets = {
        secret
        for row in rows
        for secret in (
            [str(row["api_key"] or "").strip()]
            + decode_api_keys(row["api_keys_json"])
        )
        if secret
    }

    events: list[str] = []
    selected: dict[str, str] = {}

    def remember_route(endpoint: str, model: str, api_key: str) -> None:
        selected.update(endpoint=endpoint, model=model, api_key=api_key)

    started_at = time.perf_counter()
    try:
        await digester._chat(
            "You are a connectivity probe. Reply with the single word: pong.",
            "ping",
            endpoint=str(primary["endpoint"]),
            model=str(primary["model"]),
            api_key=str(primary["api_key"] or ""),
            log=events.append,
            label="Model route test",
            max_tokens=256,
            enable_skills=False,
            disable_thinking=True,
            route_selected=remember_route,
        )
    except Exception as exc:  # noqa: BLE001 - return diagnostics to Admin
        safe_events = [
            redact_api_keys(event, secrets)
            for event in events
            if _safe_route_event(event)
        ]
        return ProviderRouteTestResponse(
            ok=False,
            message=redact_api_keys(
                str(exc) or exc.__class__.__name__,
                secrets,
            ),
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            events=safe_events[-20:],
        )

    selected_endpoint = selected.get("endpoint", "")
    selected_model = selected.get("model", "")
    selected_key = selected.get("api_key", "")
    provider = next(
        (
            row
            for row in rows
            if model_router.endpoint_identity(str(row["endpoint"]))
            == model_router.endpoint_identity(selected_endpoint)
            and str(row["model"]) == selected_model
        ),
        None,
    )
    role = str(provider["route_role"]) if provider is not None else "standalone"
    return ProviderRouteTestResponse(
        ok=True,
        message="Configured route returned a response",
        route_role=role,
        provider_name=str(provider["name"]) if provider is not None else "Selected provider",
        model=selected_model,
        key_id=key_identifier(selected_key) if selected_key else None,
        fallback_used=role == "backup",
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        events=[
            redact_api_keys(event, secrets)
            for event in events
            if _safe_route_event(event)
        ][-20:],
    )


def _safe_route_event(event: str) -> bool:
    folded = event.casefold()
    return any(
        marker in folded
        for marker in (
            "model route",
            "api key",
            "rate gate",
            "switching to",
            "completed via",
            "responded in",
        )
    )
@router.put("/{provider_id}", response_model=ProviderResponse)
async def edit_provider(provider_id: int, body: ProviderUpdate):
    updates = body.model_dump(exclude_unset=True)
    # The edit form never receives the stored secret. A blank field means
    # "leave unchanged", matching the UI copy and the pre-save test behaviour.
    if not updates.get("api_key"):
        updates.pop("api_key", None)
    try:
        provider = await database.update_provider(provider_id, **updates)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
