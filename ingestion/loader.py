"""
L.O.K.I — ingestion.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Carregadores de documentos por tipo de arquivo.

Suporta: ``.pdf``, ``.txt``, ``.md``, ``.py``, ``.csv``, ``.json``.
Novos formatos podem ser adicionados registrando-se um loader
no dicionário ``LOADER_MAP``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPowerPointLoader,
)

# ════════════════════════════════════════════════════════════
#  Registry de Loaders por extensão
# ════════════════════════════════════════════════════════════

LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md":  UnstructuredMarkdownLoader,
    ".py":  TextLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
}

# Extensões de texto puro que usam TextLoader genérico
_PLAIN_TEXT_EXTS = {".txt", ".py", ".log", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".rst"}


def _get_loader(file_path: Path) -> object:
    """Retorna uma instância do loader apropriado para o arquivo.

    Raises:
        ValueError: Extensão não suportada.
    """
    ext = file_path.suffix.lower()

    if ext in LOADER_MAP:
        loader_cls = LOADER_MAP[ext]
        if loader_cls == TextLoader:
            return loader_cls(str(file_path), encoding="utf-8")
        return loader_cls(str(file_path))

    if ext in _PLAIN_TEXT_EXTS:
        return TextLoader(str(file_path), encoding="utf-8")

    raise ValueError(
        f"Extensão '{ext}' não suportada. "
        f"Suportadas: {sorted(set(list(LOADER_MAP.keys()) + list(_PLAIN_TEXT_EXTS)))}"
    )


# ════════════════════════════════════════════════════════════
#  API Pública
# ════════════════════════════════════════════════════════════

def load_single_file(file_path: Path) -> List[Document]:
    """Carrega um único arquivo e retorna os ``Document``s brutos.

    Args:
        file_path: Caminho absoluto ou relativo para o arquivo.

    Returns:
        Lista de ``Document`` (pode conter múltiplas páginas em PDFs).
    """
    loader = _get_loader(file_path)
    docs: List[Document] = loader.load()  # type: ignore[union-attr]
    # Injeta metadado padronizado
    for doc in docs:
        doc.metadata["source_file"] = str(file_path)
        doc.metadata["file_type"] = file_path.suffix.lower()
    return docs


def load_path(target: str | Path) -> List[Document]:
    """Carrega um arquivo ou todos os arquivos suportados de um diretório.

    Args:
        target: Caminho para arquivo ou diretório.

    Returns:
        Lista agregada de ``Document``.

    Raises:
        FileNotFoundError: Caminho inexistente.
    """
    target = Path(target).resolve()

    if not target.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {target}")

    if target.is_file():
        return load_single_file(target)

    # ── Diretório: coleta recursiva ─────────────────────────
    all_docs: List[Document] = []
    supported_exts = set(LOADER_MAP.keys()) | _PLAIN_TEXT_EXTS
    files: Sequence[Path] = sorted(
        f for f in target.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_exts
    )

    if not files:
        raise ValueError(
            f"Nenhum arquivo suportado encontrado em: {target}"
        )

    for file in files:
        try:
            all_docs.extend(load_single_file(file))
        except Exception as e:
            print(f"\n[Aviso] Falha ao carregar o arquivo '{file.name}': {e}")
            continue

    return all_docs
