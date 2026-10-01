"""
L.O.K.I — config.env_setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validação e injeção proativa de variáveis de ambiente para
aceleração de GPU AMD RDNA 3 (ROCm) antes de qualquer chamada
ao Ollama ou LangChain.

Este módulo DEVE ser importado antes de qualquer outro módulo
que interaja com o LLM. A função ``setup_gpu_environment()`` é
chamada automaticamente na importação do módulo.

Variáveis injetadas:
    - ``OLLAMA_FLASH_ATTENTION=1``
      Ativa Flash Attention para otimizar uso de VRAM e contexto.

    - ``OLLAMA_KEEP_ALIVE=-1``
      Impede o descarregamento automático do modelo da VRAM,
      mantendo-o quente entre chamadas.

    - ``OLLAMA_NUM_PARALLEL=4``
      Permite até 4 chamadas simultâneas de extração, acelerando
      a ingestão chunk-a-chunk.

    - ``HSA_OVERRIDE_GFX_VERSION=11.0.0``
      Força o reconhecimento do chip gfx1100 (AMD RX 7900 XT,
      arquitetura RDNA 3) pelo runtime ROCm/HIP.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger("loki.env_setup")


# ════════════════════════════════════════════════════════════
#  Definição das variáveis de ambiente para GPU AMD
# ════════════════════════════════════════════════════════════

_GPU_ENV_VARS: List[Tuple[str, str, str]] = [
    ("OLLAMA_FLASH_ATTENTION", "1",       "Flash Attention (VRAM otimizada)"),
    ("OLLAMA_KEEP_ALIVE",     "-1",      "Modelo permanente na VRAM"),
    ("OLLAMA_NUM_PARALLEL",   "1",       "Chamadas paralelas ao LLM (1 p/ estabilidade na RX 7900 XT)"),
    ("HSA_OVERRIDE_GFX_VERSION", "11.0.0", "Override ROCm -> gfx1100 (RDNA 3)"),
]


# ════════════════════════════════════════════════════════════
#  Função principal de setup
# ════════════════════════════════════════════════════════════

def setup_gpu_environment() -> Dict[str, str]:
    """Valida e injeta variáveis de ambiente para aceleração GPU AMD.

    As variáveis só são definidas se ainda não existirem no ambiente,
    permitindo override manual pelo usuário via ``.env`` ou shell.

    Returns:
        Dict com as variáveis efetivamente aplicadas e seus valores.
    """
    applied: Dict[str, str] = {}

    for var_name, default_value, description in _GPU_ENV_VARS:
        current = os.environ.get(var_name)

        if current is None:
            # Variável não definida — injetar o default
            os.environ[var_name] = default_value
            applied[var_name] = default_value
            logger.debug(
                "GPU env injetada: %s=%s (%s)",
                var_name, default_value, description,
            )
        else:
            # Variável já definida pelo usuário — preservar
            applied[var_name] = current
            logger.debug(
                "GPU env existente: %s=%s (%s) [preservada]",
                var_name, current, description,
            )

    return applied


def get_gpu_status_report() -> str:
    """Gera um relatório legível do estado das variáveis de GPU.

    Returns:
        String formatada com o status de cada variável.
    """
    lines: List[str] = []
    for var_name, default_value, description in _GPU_ENV_VARS:
        current = os.environ.get(var_name, "NÃO DEFINIDA")
        is_default = current == default_value
        status = "[OK]" if current != "NÃO DEFINIDA" else "[--]"
        source = "(default)" if is_default else "(custom)"
        lines.append(f"  {status} {var_name}={current} {source} - {description}")

    return "\n".join(lines)


def validate_rocm_compatibility() -> bool:
    """Verifica se o ambiente parece compatível com ROCm.

    Checa indicadores comuns de que o ROCm está disponível:
    - Presença de ``HSA_OVERRIDE_GFX_VERSION`` no ambiente
    - Existência do diretório ``/opt/rocm`` (Linux) ou variável ``HIP_PATH``

    Returns:
        True se indicadores ROCm forem encontrados.
    """
    # Windows: verificar HIP_PATH
    hip_path = os.environ.get("HIP_PATH")
    if hip_path and os.path.isdir(hip_path):
        return True

    # Linux: verificar /opt/rocm
    if os.path.isdir("/opt/rocm"):
        return True

    # Fallback: se HSA_OVERRIDE já está setada, assumir compatível
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return True

    return False


# ════════════════════════════════════════════════════════════
#  Auto-execução na importação
# ════════════════════════════════════════════════════════════

# Injetar variáveis imediatamente ao importar este módulo.
# Isso garante que o ambiente esteja preparado antes de
# qualquer instanciação do ChatOllama ou LLMGraphTransformer.
_applied_vars = setup_gpu_environment()
