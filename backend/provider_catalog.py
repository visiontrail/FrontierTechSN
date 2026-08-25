"""Known model-provider presets exposed by the Admin Models page.

The provider/model associations mirror RavenAIService's Admin model catalogue
(``app/agents/anthropic_client.py`` at 0f1d23d793e892e7387690a50276828eb4c6bfcf).
FrontierTechSN keeps the catalogue separate from provider rows so multiple
accounts or gateways can use the same provider type.  It intentionally contains
no primary/backup slot, failover, circuit-breaker, or routing policy state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProviderProfile:
    id: str
    label: str
    default_endpoint: str
    default_model: str
    models: tuple[str, ...]
    notes: str

    @property
    def endpoint_needs_input(self) -> bool:
        return bool(re.search(r"\{[^{}]+\}", self.default_endpoint))

    def describe(self) -> dict[str, object]:
        data = asdict(self)
        data["models"] = list(self.models)
        data["endpoint_needs_input"] = self.endpoint_needs_input
        return data


PROVIDER_PROFILES: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        id="anthropic",
        label="Anthropic 官方",
        default_endpoint="https://api.anthropic.com",
        default_model="claude-sonnet-4-6",
        models=(
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ),
        notes="Anthropic 官方端点。",
    ),
    ProviderProfile(
        id="deepseek",
        label="DeepSeek 深度求索",
        default_endpoint="https://api.deepseek.com/anthropic",
        default_model="deepseek-v4-pro",
        models=("deepseek-v4-pro", "deepseek-v4-flash"),
        notes="DeepSeek Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="aliyun",
        label="阿里云百炼 / 通义千问",
        default_endpoint="https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/apps/anthropic",
        default_model="qwen3.7-max",
        models=(
            "qwen3.7-max",
            "qwen3.7-plus",
            "qwen3.7-flash",
            "qwen3-coder-next",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
            "qwen3-vl-plus",
            "qwen3-vl-flash",
            "qwen3.6-27b",
        ),
        notes="将端点中的 {WorkspaceId} 替换为百炼工作空间 ID。",
    ),
    ProviderProfile(
        id="zhipu",
        label="智谱 AI / GLM",
        default_endpoint="https://open.bigmodel.cn/api/anthropic",
        default_model="glm-5.2",
        models=("glm-5.2",),
        notes="智谱 GLM Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="moonshot",
        label="月之暗面 / Kimi",
        default_endpoint="https://api.moonshot.cn/anthropic",
        default_model="kimi-k3",
        models=(
            "kimi-k3",
            "kimi-k2.7-code",
            "kimi-k2.7-code-highspeed",
            "kimi-k2.6",
        ),
        notes="Kimi Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="minimax",
        label="MiniMax 稀宇科技",
        default_endpoint="https://api.minimaxi.com/anthropic",
        default_model="MiniMax-M3",
        models=("MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5"),
        notes="MiniMax Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="stepfun",
        label="阶跃星辰 StepFun",
        default_endpoint="https://api.stepfun.com",
        default_model="step-3.7-flash",
        models=("step-3.7-flash", "step-3.5-flash-2603", "step-3.5-flash"),
        notes="StepFun Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="mimo",
        label="小米 MiMo",
        default_endpoint="https://api.xiaomimimo.com/anthropic",
        default_model="mimo-v2.5-pro",
        models=("mimo-v2.5-pro", "mimo-v2.5"),
        notes="小米 MiMo Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="hunyuan",
        label="腾讯混元",
        default_endpoint="https://api.hunyuan.cloud.tencent.com/anthropic",
        default_model="hunyuan-2.0-thinking-20251109",
        models=(
            "hunyuan-2.0-thinking-20251109",
            "hunyuan-2.0-instruct-20251111",
        ),
        notes="腾讯混元 Anthropic 兼容端点。",
    ),
    ProviderProfile(
        id="yinhe",
        label="银河内部模型（OneAPI）",
        default_endpoint="http://oneapi.yhroot.com",
        default_model="yinhe-thinking",
        models=("yinhe-thinking", "yinhe-chat"),
        notes="公司内部 OneAPI 网关。",
    ),
    ProviderProfile(
        id="custom",
        label="自定义 Anthropic 兼容端点",
        default_endpoint="",
        default_model="",
        models=(),
        notes="手动填写端点与模型 ID。",
    ),
)

PROVIDER_PROFILE_MAP = {profile.id: profile for profile in PROVIDER_PROFILES}
PROVIDER_TYPES = frozenset(PROVIDER_PROFILE_MAP)


def describe_provider_catalog() -> list[dict[str, object]]:
    return [profile.describe() for profile in PROVIDER_PROFILES]


def infer_provider_type(endpoint: str, model: str) -> str:
    """Infer catalogue metadata for legacy rows without changing runtime config."""
    endpoint_host = urlsplit(endpoint).netloc.casefold()
    if not endpoint_host:
        return "custom"
    for profile in PROVIDER_PROFILES:
        if profile.id == "custom" or model not in profile.models:
            continue
        if urlsplit(profile.default_endpoint).netloc.casefold() == endpoint_host:
            return profile.id
    return "custom"
