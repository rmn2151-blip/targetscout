"""Shared LLM helper: free OpenAI-compatible endpoint, or Anthropic/OpenAI."""
from __future__ import annotations
import os

from targetscout.config import settings


def complete(prompt: str, max_tokens: int = 800) -> str:
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "x"))
        r = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama3.1"),
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        return r.choices[0].message.content
    env = settings()["env"]
    if env.get("anthropic_api_key"):
        import anthropic
        client = anthropic.Anthropic(api_key=env["anthropic_api_key"])
        msg = client.messages.create(model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    if env.get("openai_api_key"):
        from openai import OpenAI
        client = OpenAI(api_key=env["openai_api_key"])
        r = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        return r.choices[0].message.content
    raise RuntimeError("No LLM configured. Set LLM_BASE_URL or an API key in .env")
