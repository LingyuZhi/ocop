import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelConfig(StrictModel):
    provider: Literal["deepseek", "openrouter"]
    requested_model: str
    model_id: str = Field(min_length=1)
    verified_model_id: str | None
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    thinking: bool | None = None
    reasoning_effort: Literal["low", "high", "max"] | None = None
    temperature: float | None = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    max_output_tokens: int = Field(gt=0)
    answer_format: str | None = None
    provider_order: list[str] | None = None

    @model_validator(mode="after")
    def check_provider_parameters(self):
        if self.verified_model_id is not None and self.verified_model_id != self.model_id:
            raise ValueError("verified_model_id must match model_id")
        if self.provider == "deepseek":
            if self.thinking is not True or self.reasoning_effort is None:
                raise ValueError("DeepSeek requires explicit thinking and reasoning_effort")
            if self.temperature is not None or self.top_p < 0.95:
                raise ValueError("DeepSeek thinking requires temperature=null and top_p>=0.95")
        elif self.thinking is not None or self.reasoning_effort is not None:
            raise ValueError("This OpenRouter adapter uses the non-thinking worker model")
        return self

    @property
    def base_url(self) -> str:
        return {"deepseek": "https://api.deepseek.com", "openrouter": "https://openrouter.ai/api/v1"}[self.provider]

    def completion_body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {
            "model": self.model_id, "messages": messages, "stream": False,
            "top_p": self.top_p, "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.provider == "deepseek":
            body.update(thinking={"type": "enabled"}, reasoning_effort=self.reasoning_effort)
        else:
            body["provider"] = {"allow_fallbacks": False, "require_parameters": True}
            if self.provider_order:
                body["provider"]["order"] = self.provider_order
        return body


class RequestConfig(StrictModel):
    concurrency: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(gt=0)
    consecutive_exhausted_request_limit: int = Field(gt=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)


class TrainingRuntime(StrictModel):
    microbatch_size: Literal[1]
    gradient_accumulation: int = Field(gt=0)
    betas: list[float] = Field(min_length=2, max_length=2)
    epsilon: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    scheduler: Literal["constant"]
    max_grad_norm: float = Field(gt=0)
    loss_chunk_tokens: int = Field(gt=0)
    attention: Literal["sdpa"]
    precision: Literal["bf16 parameters and Adam moments; fp32 cross entropy"]
    shuffle: Literal[True]
    checkpoint_every_steps: int = Field(gt=0)
    checkpoint_retention: Literal["all"]
    generation_max_new_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def check_betas(self):
        if any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        return self


class RuntimeConfig(StrictModel):
    config_version: Literal["ocop.prototype.v1"]
    contract_version: Literal["ocop.graph.v1"]
    seed: int
    artifacts_dir: str
    metrics_backend: Literal["tensorboard"]
    storage: Literal["sqlite_and_files"]
    benchmark: dict[str, Any]
    collection: dict[str, Any]
    requests: RequestConfig
    strong_model: ModelConfig
    worker_model: ModelConfig
    finalizer: ModelConfig
    policy: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    diagnostics: dict[str, Any]
    training_runtime: TrainingRuntime | None = None

    @model_serializer(mode="wrap")
    def serialize_optional_training(self, handler):
        result = handler(self)
        if self.training_runtime is None:
            result.pop("training_runtime", None)
        return result


def load_config(path: Path) -> RuntimeConfig:
    return RuntimeConfig.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
