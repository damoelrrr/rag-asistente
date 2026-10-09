"""Fábrica del LLM. Cambia de proveedor con LLM_PROVIDER=anthropic|openai|groq."""
from __future__ import annotations

import os
from .config import Settings


def get_chat_model(settings: Settings, judge: bool = False, max_tokens: int = 700):
    provider = (settings.judge_provider if judge and settings.judge_provider else settings.llm_provider).lower()
    model = (settings.judge_model if judge and settings.judge_model else settings.llm_model) or settings.default_model(provider)
    temperature = 0.0 if judge else settings.temperature
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)
        
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max(max_tokens, 1500),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        
    raise ValueError(f"LLM_PROVIDER desconocido: {provider!r} (usa 'anthropic', 'openai' o 'groq')")


def text_of(content) -> str:
    """Normaliza el contenido de un mensaje (str o lista de bloques) a texto."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
