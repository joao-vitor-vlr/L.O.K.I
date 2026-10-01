#!/usr/bin/env python3
"""
L.O.K.I — main.py
~~~~~~~~~~~~~~~~~~~
Ponto de entrada do aplicativo "Segundo Cérebro" Graph RAG CLI.

Uso:
    python main.py

Pré-requisitos:
    1. Instalar dependências: ``pip install -r requirements.txt``
    2. Copiar ``.env.example`` para ``.env`` e configurar.
    3. Se usar Ollama local, garantir que o servidor esteja rodando.
"""

from __future__ import annotations

import sys

from config.settings import settings
from cli.interface import LokiCLI, console


def main() -> None:
    """Inicializa e executa o L.O.K.I."""
    try:
        app = LokiCLI(settings)
        app.run()
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Até logo! 🧠[/bold cyan]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Erro fatal: {e}[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
