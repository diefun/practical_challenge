"""Chat model factory (Claude via langchain-anthropic)."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from agent.config import get_config


def make_llm(effort: str | None = None) -> ChatAnthropic:
    """Return a Claude chat model.

    Notes for claude-sonnet-5-5:
    * thinking is adaptive by default and cannot be set to "disabled"; depth is tuned with effort.
    * forced tool_choice ("any"/"tool") is rejected, so tool selection is left to the model (auto).
    """
    cfg = get_config()["llm"]
    kwargs: dict = {}
    if cfg.get("refusal_fallbacks"):
        # Server-side fallback: if the model declines, the API retries on a suitable model.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["model_kwargs"] = {"fallbacks": "default"}
    return ChatAnthropic(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        output_config={"effort": effort or cfg["supervisor_effort"]},
        **kwargs,
    )
