"""
L.O.K.I — graph.graph_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Motor de Extração e Gerenciamento do Grafo de Conhecimento.

Este módulo contém duas classes complementares:

- ``GraphManager``: Motor principal que encapsula o LLM
  (``ChatOllama``), o ``LLMGraphTransformer`` e o ``DiGraph``
  do ``networkx``. Responsável por extrair entidades/relações
  dos documentos e persistir o grafo em disco.

- ``KnowledgeGraph``: Interface de consulta e visualização
  sobre o grafo persistido. Usada pelo módulo de consulta
  (``rag.chain``) para recuperar contexto baseado na topologia.

Dependências:
    - ``networkx`` — Manipulação de grafos direcionados
    - ``pyvis`` — Renderização visual em HTML interativo
    - ``langchain_ollama`` — ChatOllama para LLM local
    - ``langchain_experimental`` — LLMGraphTransformer
    - ``langchain_core`` — Document, BaseChatModel
"""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from config.settings import Settings

logger = logging.getLogger("loki.graph_manager")


# ════════════════════════════════════════════════════════════
#  Taxonomia Controlada — Evitar alucinação de tipos
# ════════════════════════════════════════════════════════════

ALLOWED_NODE_TYPES: List[str] = [
    "Conceito",
    "Tecnologia",
    "Pessoa",
    "Projeto",
    "Ferramenta",
    "Evento",
]

ALLOWED_RELATIONSHIP_TYPES: List[str] = [
    "UTILIZA",
    "DEPENDE_DE",
    "CRIOU",
    "RESOLVE",
    "CONECTA_SE",
    "RELACIONADO_A",
]


# ════════════════════════════════════════════════════════════
#  GraphManager — Motor de Extração e Persistência
# ════════════════════════════════════════════════════════════

class GraphManager:
    """Motor principal do Graph RAG.

    Encapsula o ciclo completo: LLM → extração de entidades →
    inserção no grafo → persistência em disco.

    Attributes:
        graph: Grafo direcionado ``networkx.DiGraph``.
        llm: Instância do LLM para extração de entidades.
        transformer: ``LLMGraphTransformer`` configurado com
            taxonomia controlada.
        settings: Configurações do projeto.
    """

    def __init__(self, llm: BaseChatModel, settings: Settings) -> None:
        self.settings: Settings = settings
        self._persist_path: Path = Path(settings.graph_persist_path).resolve()
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        # ── LLM e Transformer ──────────────────────────────
        self.llm: BaseChatModel = llm
        self.transformer = self._build_transformer()

        # ── Grafo ───────────────────────────────────────────
        self.graph: nx.DiGraph = self._load_graph()

        logger.info(
            "GraphManager inicializado: %d nós, %d arestas",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ── Builder do Transformer ──────────────────────────────

    def _build_transformer(self) -> object:
        """Instancia o ``LLMGraphTransformer`` com taxonomia controlada.

        Returns:
            Instância configurada do transformer.

        Raises:
            ImportError: Se ``langchain_experimental`` não estiver instalado.
        """
        from langchain_experimental.graph_transformers import LLMGraphTransformer

        transformer = LLMGraphTransformer(
            llm=self.llm,
            additional_instructions=(
                "Sempre que possível, crie nós de grandes categorias (macrogrupos) e "
                "relacione os itens menores a eles. Por exemplo, se encontrar nomes de "
                "países, crie um nó genérico 'País' e conecte-os com 'PERTENCE_A'. "
                "Faça isso para qualquer domínio (Receitas, Ingredientes, Ferramentas, etc), "
                "garantindo que itens soltos fiquem ancorados em um grupo maior."
            )
        )

        logger.debug(
            "LLMGraphTransformer configurado em MODO ABERTO (Universal) "
            "— sem restrições de taxonomia."
        )

        return transformer

    # ── Persistência ────────────────────────────────────────

    def _load_graph(self) -> nx.DiGraph:
        """Carrega o grafo do disco ou cria um novo.

        Trata arquivos corrompidos retornando um grafo vazio
        e logando o erro para diagnóstico.

        Returns:
            ``nx.DiGraph`` carregado ou vazio.
        """
        if not self._persist_path.exists():
            logger.info("Arquivo de grafo não encontrado — criando novo DiGraph")
            return nx.DiGraph()

        if self._persist_path.stat().st_size == 0:
            logger.warning("Arquivo de grafo vazio — criando novo DiGraph")
            return nx.DiGraph()

        try:
            graph = nx.read_graphml(str(self._persist_path))
            logger.info(
                "Grafo carregado de %s (%d nós, %d arestas)",
                self._persist_path.name,
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return graph
        except Exception as e:
            logger.error(
                "Falha ao carregar grafo de %s: %s — criando novo DiGraph",
                self._persist_path, e,
            )
            # Backup do arquivo corrompido antes de sobrescrever
            backup_path = self._persist_path.with_suffix(".graphml.bak")
            try:
                self._persist_path.rename(backup_path)
                logger.info("Arquivo corrompido movido para %s", backup_path)
            except OSError:
                pass
            return nx.DiGraph()

    def _save_graph(self) -> None:
        """Persiste o grafo atual em disco no formato GraphML.

        Raises:
            OSError: Se não for possível escrever no disco.
        """
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            nx.write_graphml(self.graph, str(self._persist_path))
            logger.debug(
                "Grafo salvo em %s (%d nós, %d arestas)",
                self._persist_path.name,
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )
        except OSError as e:
            logger.error("Falha ao salvar grafo: %s", e)
            raise

    # ── Ingestão Pública ────────────────────────────────────

    def process_and_ingest(
        self,
        text_chunks: List[str],
        source_name: str,
        on_chunk_progress: Optional[callable] = None,
    ) -> Tuple[int, int]:
        """Extrai entidades/relações dos chunks e insere no grafo.

        Pipeline:
        1. Converte ``text_chunks`` em objetos ``Document``.
        2. Passa os documentos no ``LLMGraphTransformer``.
        3. Itera sobre os resultados inserindo nós e arestas.
        4. Persiste o grafo no disco.

        Args:
            text_chunks: Lista de strings com os blocos de texto.
            source_name: Nome do arquivo de origem (metadado).
            on_chunk_progress: Callback ``(current, total)`` opcional.

        Returns:
            Tupla ``(nós_inseridos, arestas_inseridas)``.

        Raises:
            ConnectionError: Se o LLM não estiver acessível.
        """
        total_nodes_added: int = 0
        total_edges_added: int = 0
        total_chunks: int = len(text_chunks)

        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                if on_chunk_progress:
                    on_chunk_progress(i + 1, total_chunks)
                continue

            # ── Converter para Document ──────────────────────
            doc = Document(
                page_content=chunk_text,
                metadata={"source": source_name, "chunk_index": i},
            )

            # ── Extrair entidades e relações via LLM ─────────
            try:
                graph_documents = self.transformer.convert_to_graph_documents([doc])
            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    "Falha na extração do chunk %d/%d: %s",
                    i + 1, total_chunks, error_msg,
                )
                # Abortar se for erro de conexão
                if any(kw in error_msg for kw in (
                    "Connection", "Max retries", "ConnectError",
                    "refused", "Timeout", "timeout",
                )):
                    raise ConnectionError(
                        f"Erro de conexão com o LLM no chunk {i + 1}: {error_msg}"
                    ) from e
                # Outros erros: pular o chunk
                if on_chunk_progress:
                    on_chunk_progress(i + 1, total_chunks)
                continue

            # ── Inserir nós e arestas no grafo ───────────────
            for graph_doc in graph_documents:
                # Nós
                for node in graph_doc.nodes:
                    node_id = node.id.strip().upper()
                    node_type = node.type if node.type else "Conceito"

                    if not node_id:
                        continue

                    if not self.graph.has_node(node_id):
                        self.graph.add_node(
                            node_id,
                            label=node_id,
                            entity_type=node_type,
                            source=source_name,
                        )
                        total_nodes_added += 1
                    else:
                        # Atualizar fontes no nó existente
                        existing_sources = self.graph.nodes[node_id].get("source", "")
                        if source_name not in existing_sources:
                            updated = f"{existing_sources}|{source_name}" if existing_sources else source_name
                            self.graph.nodes[node_id]["source"] = updated

                # Arestas
                for rel in graph_doc.relationships:
                    src = rel.source.id.strip().upper()
                    tgt = rel.target.id.strip().upper()
                    rel_type = rel.type.strip().upper() if rel.type else "RELACIONADO_A"

                    if not src or not tgt:
                        continue

                    # Garantir que os nós existam
                    for nid in (src, tgt):
                        if not self.graph.has_node(nid):
                            self.graph.add_node(
                                nid,
                                label=nid,
                                entity_type="Conceito",
                                source=source_name,
                            )
                            total_nodes_added += 1

                    if not self.graph.has_edge(src, tgt):
                        self.graph.add_edge(
                            src, tgt,
                            relation=rel_type,
                            label=rel_type,
                            source=source_name,
                        )
                        total_edges_added += 1
                    else:
                        # Se a aresta existe com relação diferente, concatenar
                        existing_rel = self.graph.edges[src, tgt].get("relation", "")
                        if rel_type not in existing_rel:
                            combined = f"{existing_rel}|{rel_type}" if existing_rel else rel_type
                            self.graph.edges[src, tgt]["relation"] = combined
                            self.graph.edges[src, tgt]["label"] = combined

            if on_chunk_progress:
                on_chunk_progress(i + 1, total_chunks)

        # ── Persistir ────────────────────────────────────────
        self._save_graph()

        logger.info(
            "Ingestão '%s': +%d nós, +%d arestas (total: %d nós, %d arestas)",
            source_name,
            total_nodes_added,
            total_edges_added,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return total_nodes_added, total_edges_added

    # ── Estatísticas ────────────────────────────────────────

    @property
    def node_count(self) -> int:
        """Número total de entidades (nós) no grafo."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Número total de relações (arestas) no grafo."""
        return self.graph.number_of_edges()


# ════════════════════════════════════════════════════════════
#  KnowledgeGraph — Interface de Consulta e Visualização
# ════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """Interface de consulta e visualização sobre o grafo.

    Opera sobre o mesmo ``DiGraph`` do ``GraphManager``, expondo
    métodos de busca de vizinhança, contexto para RAG e
    renderização HTML via ``pyvis``.

    Pode ser inicializada com um ``GraphManager`` existente
    (compartilhando o grafo) ou diretamente de um arquivo GraphML.

    Attributes:
        graph: Referência ao ``DiGraph`` do grafo de conhecimento.
        settings: Configurações do projeto.
    """

    def __init__(
        self,
        settings: Settings,
        graph_manager: Optional[GraphManager] = None,
    ) -> None:
        self.settings: Settings = settings

        if graph_manager is not None:
            # Compartilhar o grafo do GraphManager (mesma referência)
            self.graph: nx.DiGraph = graph_manager.graph
        else:
            # Carregamento standalone a partir do disco
            persist_path = Path(settings.graph_persist_path).resolve()
            if persist_path.exists() and persist_path.stat().st_size > 0:
                try:
                    self.graph = nx.read_graphml(str(persist_path))
                except Exception:
                    self.graph = nx.DiGraph()
            else:
                self.graph = nx.DiGraph()

    # ── Consultas no Grafo ──────────────────────────────────

    def get_entity_neighborhood(
        self,
        entity: str,
        depth: int = 2,
    ) -> List[Tuple[str, str, str]]:
        """Retorna as trincas na vizinhança de uma entidade.

        Args:
            entity: Nome da entidade (case-insensitive).
            depth: Profundidade da busca BFS.

        Returns:
            Lista de ``(sujeito, relação, objeto)`` na vizinhança.
        """
        entity_upper = entity.strip().upper()
        if entity_upper not in self.graph:
            return []

        # BFS para encontrar nós na vizinhança
        neighborhood: Set[str] = set()
        current_layer: Set[str] = {entity_upper}

        for _ in range(depth):
            next_layer: Set[str] = set()
            for node in current_layer:
                neighborhood.add(node)
                next_layer.update(self.graph.successors(node))
                next_layer.update(self.graph.predecessors(node))
            current_layer = next_layer - neighborhood

        neighborhood.update(current_layer)

        # Coletar trincas do subgrafo
        triplets: List[Tuple[str, str, str]] = []
        subgraph = self.graph.subgraph(neighborhood)
        for u, v, data in subgraph.edges(data=True):
            relation = data.get("relation", "RELACIONADO_A")
            triplets.append((u, relation, v))

        return triplets

    def search_entities(self, query: str) -> List[str]:
        """Busca entidades no grafo cujo nome contenha o termo.

        Args:
            query: Termo de busca (case-insensitive).

        Returns:
            Lista de nomes de entidades encontradas.
        """
        query_upper = query.strip().upper()
        return [
            node for node in self.graph.nodes
            if query_upper in node
        ]

    def get_context_for_query(self, query: str, depth: int = 2) -> str:
        """Gera contexto textual baseado no grafo para uma query.

        Extrai palavras-chave da query, busca entidades correspondentes
        e retorna as trincas formatadas como contexto.

        Args:
            query: Pergunta do usuário.
            depth: Profundidade da vizinhança.

        Returns:
            Contexto formatado como texto legível.
        """
        # Tokenização simples: palavras com 3+ caracteres
        keywords = [
            w for w in query.split()
            if len(w) >= 3 and w.lower() not in _STOP_WORDS_PT
        ]

        all_triplets: List[Tuple[str, str, str]] = []
        seen_entities: Set[str] = set()

        for keyword in keywords:
            matching = self.search_entities(keyword)
            for entity in matching:
                if entity not in seen_entities:
                    seen_entities.add(entity)
                    triplets = self.get_entity_neighborhood(entity, depth)
                    all_triplets.extend(triplets)

        if not all_triplets:
            return ""

        # Deduplica e formata
        unique_triplets = list(set(all_triplets))
        lines: List[str] = []
        for subj, rel, obj in sorted(unique_triplets):
            lines.append(f"  • {subj}  —[{rel}]→  {obj}")

        header = f"Relações encontradas ({len(unique_triplets)} trincas):"
        return f"{header}\n" + "\n".join(lines)

    # ── Estatísticas ────────────────────────────────────────

    @property
    def node_count(self) -> int:
        """Número total de entidades (nós) no grafo."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Número total de relações (arestas) no grafo."""
        return self.graph.number_of_edges()

    # ── Visualização ────────────────────────────────────────

    def render_html(self, output_path: Optional[str] = None) -> str:
        """Renderiza o grafo como HTML interativo usando ``gravis``.

        Nós são dimensionados pelo grau de conexão e coloridos dinamicamente
        através de um hash sobre o seu tipo de entidade (group).
        Gravis usa WebGL e D3-Force, permitindo renderizar dezenas
        de milhares de nós em 60 FPS no navegador.

        Args:
            output_path: Caminho do arquivo HTML de saída.
                         Se None, usa ``settings.graph_vis_path``.

        Returns:
            Caminho absoluto do arquivo HTML gerado.
        """
        import gravis as gv
        import hashlib

        out = Path(output_path or self.settings.graph_vis_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        # ── Calcular graus para dimensionar nós ──────────────
        degrees: Dict[str, int] = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees else 1

        def get_color_for_group(group: str) -> str:
            # Gera uma cor hexadecimal pastel consistente baseada no nome do grupo
            hash_val = int(hashlib.md5(group.encode('utf-8')).hexdigest(), 16)
            # Mistura com branco para suavizar a cor (pastel)
            r = (hash_val & 0xFF0000) >> 16
            g = (hash_val & 0x00FF00) >> 8
            b = hash_val & 0x0000FF
            r = (r + 255) // 2
            g = (g + 255) // 2
            b = (b + 255) // 2
            return f"#{r:02x}{g:02x}{b:02x}"

        # Criar grafo temporário para injeção de propriedades visuais
        vis_graph = nx.DiGraph()

        # ── Adicionar nós ────────────────────────────────────
        for node_id, data in self.graph.nodes(data=True):
            degree = degrees.get(node_id, 1)
            size = 15 + (degree / max_degree) * 35

            entity_type = data.get("entity_type", "Conceito")
            color = get_color_for_group(entity_type)

            sources = data.get("source", "")
            hover_text = (
                f"<b>{node_id}</b><br>"
                f"Tipo: {entity_type}<br>"
                f"Conexões: {degree}"
            )
            if sources:
                hover_text += f"<br>Fontes: {sources.replace('|', ', ')}"

            vis_graph.add_node(
                node_id,
                size=size,
                color=color,
                label=data.get("label", node_id),
                hover=hover_text,
                group=entity_type
            )

        # ── Adicionar arestas ────────────────────────────────
        for u, v, data in self.graph.edges(data=True):
            relation = data.get("relation", "")
            vis_graph.add_edge(
                u, v,
                label=relation,
                hover=f"<b>{u}</b> → <i>{relation}</i> → <b>{v}</b>",
                color="#4a9eff",
                size=1.0,
            )

        # ── Renderizar com Gravis (D3) ───────────────────────
        fig = gv.d3(
            vis_graph,
            graph_height=900,
            show_details=True,
            show_details_toggle_button=True,
            show_menu_toggle_button=True,
            show_node_label=True,
            show_edge_label=True,
            edge_label_data_source="label",
            node_hover_tooltip=True,
            edge_hover_tooltip=True,
            links_force_distance=150,
            use_edge_size_normalization=True,
            edge_curvature=0.2, # Para arestas direcionadas
            layout_algorithm_active=True,
            node_label_size_factor=1.2,
        )

        # Salvar HTML
        with open(out, "w", encoding="utf-8") as f:
            f.write(fig.to_html_standalone())

        return str(out)

    def open_visualization(self, output_path: Optional[str] = None) -> str:
        """Renderiza e abre o grafo no navegador padrão.

        Returns:
            Caminho do arquivo HTML gerado.
        """
        html_path = self.render_html(output_path)
        webbrowser.open(f"file://{html_path}")
        return html_path


# ════════════════════════════════════════════════════════════
#  Stop Words (PT-BR) — usadas na busca de entidades
# ════════════════════════════════════════════════════════════

_STOP_WORDS_PT: Set[str] = {
    "que", "não", "para", "por", "com", "uma", "dos", "das",
    "nos", "nas", "seu", "sua", "seus", "suas", "como", "mas",
    "foi", "são", "está", "tem", "ser", "ter", "isso", "essa",
    "esse", "isto", "esta", "este", "qual", "quais", "onde",
    "quando", "quem", "porque", "sobre", "entre", "cada",
    "mais", "muito", "também", "pode", "deve", "após", "até",
    "desde", "durante", "antes", "depois", "sem", "sob",
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "has",
    "what", "when", "who", "how", "which", "from", "with",
}
