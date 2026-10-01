"""
L.O.K.I — config.settings
~~~~~~~~~~~~~~~~~~~~~~~~~~
Carregamento centralizado de variáveis de ambiente e fábrica de
instâncias LLM via Strategy Pattern.

O módulo lê o arquivo ``.env`` na raiz do projeto e expõe um
dataclass ``Settings`` imutável, além de uma factory function
``build_llm()`` que devolve a instância correta de acordo
com o provider configurado.

IMPORTANTE: ``config.env_setup`` é importado antes de qualquer
coisa para garantir que as variáveis de GPU AMD estejam injetadas
no ambiente.
"""

from __future__ import annotations

# ── GPU Environment Setup (primeiro import) ──────────────────
import config.env_setup  # noqa: F401

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# ── Carregar .env (busca a raiz do projeto) ─────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ════════════════════════════════════════════════════════════
#  Dataclass de Configuração
# ════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Settings:
    """Configurações globais carregadas do ambiente."""

    # ── Provider ────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower()
    )

    # ── Ollama ──────────────────────────────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_llm_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
    )
    ollama_embed_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_llm_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
    )
    openai_embed_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )

    # ── Anthropic ───────────────────────────────────────────
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    anthropic_llm_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022")
    )

    # ── Graph RAG ──────────────────────────────────────────
    graph_persist_path: str = field(
        default_factory=lambda: os.getenv("GRAPH_PERSIST_PATH", "./data/knowledge_graph.graphml")
    )
    graph_vis_path: str = field(
        default_factory=lambda: os.getenv("GRAPH_VIS_PATH", "./data/mapa_loki.html")
    )

    # ── Splitter ────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )


# ════════════════════════════════════════════════════════════
#  Factory Functions (Strategy Pattern)
# ════════════════════════════════════════════════════════════

def build_llm(settings: Settings) -> BaseChatModel:
    """Instancia o LLM correto de acordo com ``settings.llm_provider``.

    Para Ollama, usa ``temperature=0`` para garantir saídas
    determinísticas na extração de entidades.

    Returns:
        BaseChatModel — instância pronta para ``.invoke()`` / ``.astream()``.

    Raises:
        ValueError: Provider desconhecido.
        ImportError: Pacote do provider não instalado.
    """
    provider = settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=8192,  # Aumenta a janela de contexto para evitar engasgos com prompts grandes
            num_predict=2048, # Garante que ele não trave gerando infinitamente
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY não configurada no .env")
        return ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY não configurada no .env")
        return ChatAnthropic(
            model=settings.anthropic_llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(
        f"Provider '{provider}' não suportado. "
        "Use: ollama | openai | anthropic"
    )


# ── Instância global (singleton simples) ───────────────────
settings = Settings()
