"""
L.O.K.I — cli.interface
~~~~~~~~~~~~~~~~~~~~~~~~
Interface de terminal interativa usando Rich + prompt_toolkit.

Responsável por:
- Renderizar o banner de boas-vindas e painéis informativos.
- Gerenciar o loop de prompt interativo com histórico.
- Despachar comandos (``!add``, ``!map``, ``!status``, texto livre → Graph RAG).
- Exibir spinners e barras de progresso durante operações longas.
"""

from __future__ import annotations

# ── GPU Environment Setup (DEVE ser o primeiro import) ──────
import config.env_setup  # noqa: F401  — injeta vars de ambiente antes de tudo

import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from rich.theme import Theme

from config.env_setup import get_gpu_status_report
from config.settings import Settings, build_llm
from graph.graph_manager import GraphManager, KnowledgeGraph
from ingestion.ingestion import ingest_to_graph
from rag.chain import build_graph_rag_chain, build_simple_chain


# ════════════════════════════════════════════════════════════
#  Tema e Console
# ════════════════════════════════════════════════════════════

_THEME = Theme({
    "info":    "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "accent":  "bold magenta",
    "gpu":     "bold yellow",
})

console = Console(theme=_THEME)

# ════════════════════════════════════════════════════════════
#  Banner
# ════════════════════════════════════════════════════════════

_BANNER = r"""
[bold cyan]
    ██╗      ██████╗ ██╗  ██╗██╗
    ██║     ██╔═══██╗██║ ██╔╝██║
    ██║     ██║   ██║█████╔╝ ██║
    ██║     ██║   ██║██╔═██╗ ██║
    ███████╗╚██████╔╝██║  ██╗██║
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝
[/bold cyan]
[dim]  ⚡ Segundo Cérebro — Graph RAG CLI (GPU Accelerated)[/dim]
"""

_HELP_TEXT = """
[bold cyan]Comandos disponíveis:[/bold cyan]

  [bold green]!add[/bold green] [dim]<arquivo_ou_pasta>[/dim]   Ingere documentos no grafo de conhecimento
  [bold green]!map[/bold green]                     Visualiza o grafo no navegador (HTML interativo)
  [bold green]!status[/bold green]                  Mostra estatísticas do grafo de conhecimento
  [bold green]!gpu[/bold green]                     Mostra status das variáveis de ambiente GPU
  [bold green]!help[/bold green]                    Exibe esta ajuda
  [bold green]sair[/bold green]                     Encerra o L.O.K.I.

  [dim]Qualquer outro texto → consulta ao Segundo Cérebro via Graph RAG[/dim]
"""


# ════════════════════════════════════════════════════════════
#  Classe Principal da CLI
# ════════════════════════════════════════════════════════════

class LokiCLI:
    """Interface interativa do L.O.K.I. com Graph RAG.

    Attributes:
        settings: Configurações globais do projeto.
        graph_manager: Motor de extração de entidades e persistência.
        knowledge_graph: Interface de consulta e visualização do grafo.
        llm: Instância do LLM ativo.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history_path = Path(".data/loki_history.txt")
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(self._history_path)),
        )

        # ── Inicialização lazy (preenchidas no boot) ────────
        self.llm = None
        self.graph_manager: Optional[GraphManager] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.rag_chain = None

    # ── Boot ────────────────────────────────────────────────

    def boot(self) -> None:
        """Inicializa LLM, GraphManager e KnowledgeGraph com feedback visual."""
        console.print(_BANNER)

        # ── GPU Environment ─────────────────────────────────
        console.print("  [gpu]⚡ GPU AMD RDNA 3 — Variáveis ROCm injetadas[/gpu]")

        # ── LLM ────────────────────────────────────────────
        with console.status(
            "[bold cyan]Conectando ao LLM...[/bold cyan]",
            spinner="dots",
        ):
            try:
                self.llm = build_llm(self.settings)
                console.print(
                    f"  [success]✔[/success] LLM: [accent]{self.settings.llm_provider}[/accent] "
                    f"({self._get_model_name()}) [dim]temperature=0[/dim]"
                )
            except Exception as e:
                console.print(f"  [error]✖ Erro ao conectar LLM: {e}[/error]")
                sys.exit(1)

        # ── GraphManager (LLM + Transformer + Grafo) ────────
        with console.status(
            "[bold cyan]Inicializando GraphManager (LLM + Transformer + Grafo)...[/bold cyan]",
            spinner="dots",
        ):
            try:
                self.graph_manager = GraphManager(
                    llm=self.llm,
                    settings=self.settings,
                )
                self.knowledge_graph = KnowledgeGraph(
                    settings=self.settings,
                    graph_manager=self.graph_manager,
                )
                n = self.knowledge_graph.node_count
                e = self.knowledge_graph.edge_count
                console.print(
                    f"  [success]✔[/success] Grafo: [accent]{n}[/accent] entidades, "
                    f"[accent]{e}[/accent] relações"
                )
            except Exception as e:
                console.print(f"  [error]✖ Erro no GraphManager: {e}[/error]")
                sys.exit(1)

        # ── RAG Chain ───────────────────────────────────────
        self._rebuild_chain()

        console.print()
        console.print(Panel(
            _HELP_TEXT,
            title="[bold]Ajuda Rápida[/bold]",
            border_style="cyan",
            padding=(0, 2),
        ))
        console.print()

    # ── Loop Principal ──────────────────────────────────────

    def run(self) -> None:
        """Executa o loop interativo de prompt."""
        self.boot()

        while True:
            try:
                user_input: str = self.session.prompt(
                    HTML("<ansicyan><b>Cérebro ❯ </b></ansicyan>"),
                ).strip()

                if not user_input:
                    continue

                # ── Comandos especiais ──────────────────────
                lower = user_input.lower()

                if lower in ("sair", "exit", "quit", "!quit"):
                    console.print(
                        "\n[bold cyan]Até logo! 🧠[/bold cyan]\n"
                    )
                    break

                if lower in ("!help", "!ajuda"):
                    console.print(Panel(
                        _HELP_TEXT,
                        title="[bold]Ajuda[/bold]",
                        border_style="cyan",
                        padding=(0, 2),
                    ))
                    continue

                if lower == "!status":
                    self._show_status()
                    continue

                if lower == "!gpu":
                    self._show_gpu_status()
                    continue

                if lower == "!map":
                    self._handle_map()
                    continue

                if lower.startswith("!add "):
                    target = user_input[5:].strip()
                    self._handle_add(target)
                    continue

                # ── Consulta Graph RAG ──────────────────────
                self._handle_query(user_input)

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'sair' para encerrar.[/dim]")
                continue
            except EOFError:
                break

    # ── Handlers ────────────────────────────────────────────

    def _handle_add(self, target: str) -> None:
        """Ingere documentos extraindo entidades e relações para o grafo."""
        path = Path(target).resolve()

        if not path.exists():
            console.print(f"  [error]✖ Caminho não encontrado: {path}[/error]")
            return

        assert self.graph_manager is not None
        assert self.knowledge_graph is not None

        console.print()
        console.print(Panel(
            f"[bold]Ingerindo:[/bold] [cyan]{path.name}[/cyan]\n"
            "[dim]Extraindo entidades e relações via LLM (taxonomia controlada)...[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ))

        # ── Progress bar para extração chunk-a-chunk ─────────
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        stats = {}

        with progress:
            task_load = progress.add_task("Carregando documentos...", total=None)
            task_split = progress.add_task("Fatiando em chunks...", total=None, visible=False)
            task_extract = progress.add_task("Extraindo entidades...", total=1, visible=False)

            def on_load(count: int) -> None:
                progress.update(task_load, completed=count, total=count)
                progress.update(
                    task_load,
                    description=f"Carregados {count} doc(s)",
                )

            def on_split(count: int) -> None:
                progress.update(task_split, visible=True, completed=count, total=count)
                progress.update(
                    task_split,
                    description=f"Fatiados em {count} chunks",
                )
                # Agora sabemos o total de chunks para a extração
                progress.update(task_extract, visible=True, total=count)

            def on_chunk_progress(current: int, total: int) -> None:
                progress.update(
                    task_extract,
                    completed=current,
                    description=f"Extraindo entidades ({current}/{total})...",
                )

            def on_extract(nodes: int, edges: int) -> None:
                progress.update(
                    task_extract,
                    description=f"Extraídos {nodes} nós, {edges} arestas",
                )

            try:
                stats = ingest_to_graph(
                    target=path,
                    graph_manager=self.graph_manager,
                    settings=self.settings,
                    on_load=on_load,
                    on_split=on_split,
                    on_extract=on_extract,
                    on_chunk_progress=on_chunk_progress,
                )
            except ConnectionError as e:
                console.print(f"\n  [error]✖ {e}[/error]\n")
                return
            except (FileNotFoundError, ValueError) as e:
                console.print(f"\n  [error]✖ {e}[/error]\n")
                return
            except Exception as e:
                console.print(f"\n  [error]✖ Erro na ingestão: {e}[/error]\n")
                return

        # ── Resumo da ingestão ───────────────────────────────
        console.print()
        summary = Text()
        summary.append("  Documentos:  ", style="bold")
        summary.append(f"{stats.get('documents_loaded', 0)}\n", style="accent")
        summary.append("  Chunks:      ", style="bold")
        summary.append(f"{stats.get('chunks_generated', 0)}\n", style="accent")
        summary.append("  Novos nós:   ", style="bold")
        summary.append(f"{stats.get('nodes_added', 0)}\n", style="accent")
        summary.append("  Novas arestas: ", style="bold")
        summary.append(f"{stats.get('edges_added', 0)}\n", style="accent")
        summary.append("  Grafo total:   ", style="bold")
        summary.append(
            f"{self.knowledge_graph.node_count} nós, "
            f"{self.knowledge_graph.edge_count} arestas",
            style="accent",
        )

        console.print(Panel(
            summary,
            title="[bold green]✔ Ingestão Completa[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

        # Rebuild chain para usar grafo atualizado
        self._rebuild_chain()

    def _handle_map(self) -> None:
        """Renderiza e abre o grafo de conhecimento no navegador."""
        assert self.knowledge_graph is not None

        if self.knowledge_graph.node_count == 0:
            console.print(
                "\n  [warning]⚠ O grafo está vazio. "
                "Use !add para ingerir documentos primeiro.[/warning]\n"
            )
            return

        with console.status(
            "[bold cyan]Gerando visualização do grafo...[/bold cyan]",
            spinner="dots",
        ):
            try:
                html_path = self.knowledge_graph.open_visualization()
            except Exception as e:
                console.print(f"\n  [error]✖ Erro na visualização: {e}[/error]\n")
                return

        console.print(
            f"\n  [success]✔[/success] Mapa gerado: [dim]{html_path}[/dim]\n"
            "  [info]Abrindo no navegador...[/info]\n"
        )

    def _handle_query(self, question: str) -> None:
        """Executa consulta Graph RAG e renderiza a resposta."""
        with console.status(
            "[bold cyan]Consultando o grafo de conhecimento...[/bold cyan]",
            spinner="dots",
        ):
            try:
                assert self.rag_chain is not None
                response: str = self.rag_chain.invoke(question)
            except Exception as e:
                console.print(
                    f"\n  [error]✖ Erro na consulta: {e}[/error]\n"
                )
                return

        # ── Renderizar resposta ─────────────────────────────
        md = Markdown(response)
        console.print()
        console.print(Panel(
            md,
            title="[bold green]L.O.K.I.[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

    def _show_status(self) -> None:
        """Exibe estatísticas do grafo de conhecimento."""
        assert self.knowledge_graph is not None

        status_text = Text()
        status_text.append("  Provider:     ", style="bold")
        status_text.append(f"{self.settings.llm_provider}\n", style="accent")
        status_text.append("  Modelo LLM:   ", style="bold")
        status_text.append(f"{self._get_model_name()}\n", style="accent")
        status_text.append("  Entidades:    ", style="bold")
        status_text.append(f"{self.knowledge_graph.node_count}\n", style="accent")
        status_text.append("  Relações:     ", style="bold")
        status_text.append(f"{self.knowledge_graph.edge_count}\n", style="accent")
        status_text.append("  Grafo (disco):", style="bold")
        status_text.append(f" {self.settings.graph_persist_path}", style="accent")

        console.print()
        console.print(Panel(
            status_text,
            title="[bold]Status do Cérebro[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

    def _show_gpu_status(self) -> None:
        """Exibe o status das variáveis de ambiente GPU AMD."""
        report = get_gpu_status_report()
        console.print()
        console.print(Panel(
            report,
            title="[bold yellow]⚡ GPU AMD RDNA 3 — ROCm Environment[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        console.print()

    # ── Internos ────────────────────────────────────────────

    def _rebuild_chain(self) -> None:
        """Reconstrói o chain Graph RAG (com ou sem dados no grafo)."""
        assert self.llm is not None
        assert self.knowledge_graph is not None

        if self.knowledge_graph.node_count > 0:
            self.rag_chain = build_graph_rag_chain(
                self.llm, self.knowledge_graph
            )
        else:
            self.rag_chain = build_simple_chain(self.llm)

    def _get_model_name(self) -> str:
        """Retorna o nome legível do modelo ativo."""
        provider = self.settings.llm_provider
        if provider == "ollama":
            return self.settings.ollama_llm_model
        if provider == "openai":
            return self.settings.openai_llm_model
        if provider == "anthropic":
            return self.settings.anthropic_llm_model
        return "desconhecido"
