"""
L.O.K.I — ingestion.ingestion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline de ingestão para Graph RAG.

Responsável por:
1. Carregar documentos de arquivo/diretório.
2. Fatiar em blocos menores com ``RecursiveCharacterTextSplitter``.
3. Delegar a extração de entidades/relações ao ``GraphManager``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings
from graph.graph_manager import GraphManager
from ingestion.loader import load_path


# ════════════════════════════════════════════════════════════
#  Fatiamento de Documentos
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
#  Pipeline Completo de Ingestão
# ════════════════════════════════════════════════════════════

def ingest_to_graph(
    target: str | Path,
    graph_manager: GraphManager,
    settings: Settings,
    on_load: Optional[Callable[[int], None]] = None,
    on_split: Optional[Callable[[int], None]] = None,
    on_extract: Optional[Callable[[int, int], None]] = None,
    on_chunk_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    """Executa o pipeline completo de ingestão Graph RAG.

    Args:
        target: Caminho do arquivo ou diretório a ingerir.
        graph_manager: Instância do ``GraphManager`` com LLM e transformer.
        settings: Configurações do projeto.
        on_load: Callback chamado após carregar documentos ``(count)``.
        on_split: Callback chamado após fatiar em chunks ``(count)``.
        on_extract: Callback chamado após extrair ``(nodes_added, edges_added)``.
        on_chunk_progress: Callback ``(current, total)`` para progresso.

    Returns:
        Dict com estatísticas da ingestão::

            {
                "documents_loaded": int,
                "chunks_generated": int,
                "nodes_added": int,
                "edges_added": int,
                "source_file": str,
            }
    """
    path = Path(target).resolve()

    # ── 1. Carregar documentos ───────────────────────────────
    documents = load_path(path)
    if on_load:
        on_load(len(documents))

    # ── 2. Fatiar em chunks ──────────────────────────────────
    chunks = split_documents(documents, settings)
    if on_split:
        on_split(len(chunks))

    # ── 3. Extrair texto dos chunks ──────────────────────────
    text_chunks: List[str] = [chunk.page_content for chunk in chunks]
    source_name = path.name

    # ── 4. Delegar ao GraphManager ───────────────────────────
    nodes_added, edges_added = graph_manager.process_and_ingest(
        text_chunks=text_chunks,
        source_name=source_name,
        on_chunk_progress=on_chunk_progress,
    )

    if on_extract:
        on_extract(nodes_added, edges_added)

    return {
        "documents_loaded": len(documents),
        "chunks_generated": len(chunks),
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "source_file": source_name,
    }
