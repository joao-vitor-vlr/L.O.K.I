"""
L.O.K.I — ingestion.splitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fatiamento de documentos e vetorização no ChromaDB.

Responsável por:
1. Fatiar ``Document``s com ``RecursiveCharacterTextSplitter``.
2. Persistir os chunks no ChromaDB via LangChain ``Chroma``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config.settings import Settings


# ════════════════════════════════════════════════════════════
#  Text Splitter
# ════════════════════════════════════════════════════════════

def split_documents(
    documents: List[Document],
    settings: Settings,
) -> List[Document]:
    """Fatia uma lista de documentos em chunks menores.

    Args:
        documents: Documentos brutos carregados pelo módulo ``loader``.
        settings: Configurações com ``chunk_size`` e ``chunk_overlap``.

    Returns:
        Lista de chunks (``Document``) com metadados preservados.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ════════════════════════════════════════════════════════════
#  Vector Store (ChromaDB)
# ════════════════════════════════════════════════════════════

class VectorStore:
    """Wrapper sobre o ChromaDB persistente via LangChain.

    Attributes:
        _store: Instância ``Chroma`` gerenciada internamente.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        settings: Settings,
    ) -> None:
        persist_dir = Path(settings.chroma_persist_dir).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        self._settings = settings

    # ── Ingestão ────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> int:
        """Adiciona documentos já fatiados ao ChromaDB.

        Args:
            documents: Chunks produzidos por ``split_documents()``.

        Returns:
            Quantidade de chunks inseridos.
        """
        if not documents:
            return 0
        self._store.add_documents(documents)
        return len(documents)

    # ── Recuperação ─────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Busca os *k* chunks mais similares à query.

        Args:
            query: Texto de busca do usuário.
            k: Quantidade de resultados (default: ``settings.rag_top_k``).

        Returns:
            Lista de ``Document`` ranqueados por similaridade.
        """
        k = k or self._settings.rag_top_k
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, k: Optional[int] = None):
        """Retorna um ``VectorStoreRetriever`` compatível com LangChain chains.

        Args:
            k: Quantidade de resultados por busca.
        """
        k = k or self._settings.rag_top_k
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    @property
    def collection_count(self) -> int:
        """Número total de documentos na coleção."""
        try:
            return self._store._collection.count()
        except Exception:
            return 0
