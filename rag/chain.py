"""
L.O.K.I — rag.chain
~~~~~~~~~~~~~~~~~~~~~
Pipeline de consulta Graph RAG.

Em vez de buscar por similaridade vetorial, este módulo:
1. Extrai entidades da pergunta do usuário.
2. Busca a vizinhança dessas entidades no grafo de conhecimento.
3. Injeta as relações (trincas) como contexto no prompt do LLM.
4. Retorna a resposta gerada pelo LLM com base na topologia do grafo.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from graph.graph_manager import KnowledgeGraph


# ════════════════════════════════════════════════════════════
#  Prompt Templates
# ════════════════════════════════════════════════════════════

_SYSTEM_TEMPLATE_GRAPH = """\
Você é o L.O.K.I., um assistente de conhecimento pessoal ("Segundo Cérebro").
Sua base de conhecimento é um **Grafo de Conhecimento** (Knowledge Graph) composto
por entidades conectadas por relações semânticas.

Abaixo estão as relações extraídas do grafo que são relevantes para a pergunta:

──────────────────────────────────────────────
CONTEXTO DO GRAFO DE CONHECIMENTO:
{context}
──────────────────────────────────────────────

Regras:
- Responda de forma clara, precisa e bem formatada em Markdown.
- Baseie sua resposta nas relações e entidades do grafo fornecidas acima.
- Se o contexto do grafo não contiver informação suficiente, diga explicitamente:
  "Não encontrei relações suficientes no grafo de conhecimento para responder."
- Explique as conexões entre entidades quando relevante.
- Nunca invente informações que não estejam no contexto do grafo.
"""

_HUMAN_TEMPLATE = "{question}"

GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_TEMPLATE_GRAPH),
    ("human", _HUMAN_TEMPLATE),
])

_SIMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é o L.O.K.I., um assistente inteligente. "
     "O grafo de conhecimento está vazio — ainda não há entidades ou relações. "
     "Responda de forma útil e sugira ao usuário adicionar documentos "
     "com o comando !add para construir o grafo."),
    ("human", "{question}"),
])


# ════════════════════════════════════════════════════════════
#  Builder dos Chains
# ════════════════════════════════════════════════════════════

def build_graph_rag_chain(
    llm: BaseChatModel,
    knowledge_graph: KnowledgeGraph,
    depth: int = 2,
) -> RunnableSerializable[str, str]:
    """Constrói o pipeline Graph RAG.

    O chain funciona da seguinte forma:
    1. Recebe a pergunta como string.
    2. Busca contexto no grafo via ``get_context_for_query()``.
    3. Monta o prompt com contexto + pergunta.
    4. Invoca o LLM e retorna a resposta como string.

    Args:
        llm: Instância do LLM (Ollama, OpenAI, Anthropic).
        knowledge_graph: Instância do ``KnowledgeGraph``.
        depth: Profundidade da busca de vizinhança no grafo.

    Returns:
        ``RunnableSerializable`` que aceita ``str`` e retorna ``str``.
    """

    def _retrieve_graph_context(question: str) -> Dict[str, str]:
        """Busca contexto no grafo e formata para o prompt."""
        context = knowledge_graph.get_context_for_query(question, depth=depth)
        if not context:
            context = "(Nenhuma entidade relevante encontrada no grafo)"
        return {"context": context, "question": question}

    chain: RunnableSerializable[str, str] = (
        RunnableLambda(_retrieve_graph_context)
        | GRAPH_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def build_simple_chain(
    llm: BaseChatModel,
) -> RunnableSerializable[str, str]:
    """Chain simplificado sem Graph RAG (para quando o grafo está vazio).

    Args:
        llm: Instância do LLM.

    Returns:
        ``RunnableSerializable`` que aceita ``str`` e retorna ``str``.
    """
    chain: RunnableSerializable[str, str] = (
        RunnableLambda(lambda q: {"question": q})
        | _SIMPLE_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
