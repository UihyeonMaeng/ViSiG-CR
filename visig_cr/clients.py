"""Local OpenAI-compatible model client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .io_utils import extract_json_object


@dataclass(frozen=True)
class JSONCompletion:
    """Parsed JSON plus the model-output fields needed for protocol audits."""

    parsed: dict[str, Any]
    raw_content: str
    finish_reason: str | None
    response_format: dict[str, Any] | None


class ModelResponseParseError(ValueError):
    """A model response was returned but did not satisfy the JSON contract."""

    def __init__(
        self,
        message: str,
        *,
        raw_content: str,
        finish_reason: str | None,
        response_format: dict[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.finish_reason = finish_reason
        self.response_format = response_format


class LocalChatClient:
    def __init__(self, cfg: dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the official openai Python package to call a local OpenAI-compatible server"
            ) from exc
        self.cfg = cfg
        client_kwargs: dict[str, Any] = {
            "base_url": str(cfg["base_url"]),
            "api_key": str(cfg.get("api_key", "EMPTY")),
            "timeout": float(cfg.get("timeout_seconds", 180)),
        }
        # The OpenAI SDK has its own transport/API retry policy.  Keep it
        # explicitly configurable so long-running local jobs can distinguish
        # model-output validation retries from HTTP/timeout retries.
        if "max_retries" in cfg:
            client_kwargs["max_retries"] = max(0, int(cfg["max_retries"]))
        self.client = OpenAI(**client_kwargs)
        self.model = str(cfg["model"])

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        allow_unconstrained_fallback: bool | None = None,
    ) -> dict[str, Any]:
        """Return one parsed JSON object.

        ``response_format`` defaults to JSON-object mode for legacy callers.
        Servers that reject response formats are *not* retried without a
        constraint unless the caller/config explicitly opts in.  In
        particular, the frozen reader always supplies a JSON Schema and keeps
        fallback disabled so a server error cannot silently weaken the output
        contract.
        """

        return self.complete_json_response(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
            response_format=response_format,
            allow_unconstrained_fallback=allow_unconstrained_fallback,
        ).parsed

    def complete_json_response(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        allow_unconstrained_fallback: bool | None = None,
    ) -> JSONCompletion:
        """Return parsed JSON and raw completion metadata for auditing."""

        max_tokens = int(max_tokens or self.cfg.get("max_tokens", 1200))
        temperature = float(
            self.cfg.get("temperature", 0.0) if temperature is None else temperature
        )
        merged_extra: dict[str, Any] = {}
        # The served API name may hide the underlying checkpoint, so honor the
        # explicit model-family profile instead of guessing from the model name.
        model_family = (
            str(self.cfg.get("model_family", ""))
            .strip()
            .lower()
            .replace("_", "-")
        )
        if (
            self.cfg.get("disable_qwen_thinking", False)
            and model_family.startswith("qwen")
        ):
            merged_extra["chat_template_kwargs"] = {"enable_thinking": False}
        if extra_body:
            merged_extra.update(extra_body)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": merged_extra or None,
        }
        selected_format = response_format or {"type": "json_object"}
        if allow_unconstrained_fallback is None:
            allow_unconstrained_fallback = bool(
                self.cfg.get("allow_unconstrained_json_fallback", False)
            )
        try:
            response = self.client.chat.completions.create(
                **kwargs,
                response_format=selected_format,
            )
        except Exception:
            if not allow_unconstrained_fallback:
                raise
            # Explicit compatibility escape hatch for non-reader legacy jobs.
            # This is intentionally opt-in and therefore never silently
            # weakens a reader's JSON Schema.
            response = self.client.chat.completions.create(**kwargs)
            selected_format = None

        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str):
            raise ValueError("Unexpected local model response type")
        finish_reason_value = getattr(response.choices[0], "finish_reason", None)
        finish_reason = (
            str(finish_reason_value) if finish_reason_value is not None else None
        )
        try:
            parsed = extract_json_object(content)
        except (TypeError, ValueError) as exc:
            raise ModelResponseParseError(
                str(exc),
                raw_content=content,
                finish_reason=finish_reason,
                response_format=selected_format,
            ) from exc
        return JSONCompletion(
            parsed=parsed,
            raw_content=content,
            finish_reason=finish_reason,
            response_format=selected_format,
        )
