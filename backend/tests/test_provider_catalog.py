import pytest
from pydantic import ValidationError

from backend.models import ProviderCreate, ProviderUpdate
from backend.provider_catalog import (
    PROVIDER_PROFILES,
    describe_provider_catalog,
    infer_provider_type,
)


def test_catalog_matches_raven_provider_order_and_model_associations():
    assert [profile.id for profile in PROVIDER_PROFILES] == [
        "anthropic",
        "deepseek",
        "aliyun",
        "zhipu",
        "moonshot",
        "minimax",
        "stepfun",
        "mimo",
        "hunyuan",
        "yinhe",
        "custom",
    ]
    profiles = {profile.id: profile for profile in PROVIDER_PROFILES}
    assert profiles["anthropic"].default_model == "claude-sonnet-4-6"
    assert profiles["deepseek"].models == ("deepseek-v4-pro", "deepseek-v4-flash")
    assert "qwen3.7-flash" in profiles["aliyun"].models
    assert profiles["moonshot"].models[0] == "kimi-k3"
    assert profiles["yinhe"].models == ("yinhe-thinking", "yinhe-chat")
    assert profiles["custom"].models == ()


def test_catalog_description_marks_tenant_endpoint_placeholders():
    catalog = {entry["id"]: entry for entry in describe_provider_catalog()}

    assert catalog["aliyun"]["endpoint_needs_input"] is True
    assert catalog["anthropic"]["endpoint_needs_input"] is False
    assert catalog["aliyun"]["models"][0] == "qwen3.7-max"
    assert not any("route" in key for entry in catalog.values() for key in entry)


def test_legacy_provider_type_inference_requires_matching_host_and_model():
    assert (
        infer_provider_type(
            "http://oneapi.yhroot.com/v1/chat/completions",
            "yinhe-thinking",
        )
        == "yinhe"
    )
    assert infer_provider_type("https://proxy.example/v1", "yinhe-thinking") == "custom"
    assert infer_provider_type("http://oneapi.yhroot.com", "new-model") == "custom"


def test_provider_payloads_accept_catalog_types_and_reject_unknown_types():
    provider = ProviderCreate(
        provider_type=" DeepSeek ",
        name="DeepSeek production",
        endpoint="https://api.deepseek.com/anthropic",
        model="deepseek-v4-pro",
    )
    update = ProviderUpdate(provider_type="YINHE")

    assert provider.provider_type == "deepseek"
    assert update.provider_type == "yinhe"
    with pytest.raises(ValidationError, match="Unknown provider type"):
        ProviderCreate(
            provider_type="not-in-the-catalog",
            name="Unknown",
            endpoint="https://example.test",
            model="model",
        )
    with pytest.raises(ValidationError, match="Replace endpoint placeholders"):
        ProviderCreate(
            provider_type="aliyun",
            name="Aliyun workspace",
            endpoint="https://{WorkspaceId}.example/apps/anthropic",
            model="qwen3.7-max",
        )
