# 🧠 L.O.K.I. — Local Operational Knowledge Interface

> **Segundo Cérebro** pessoal com **Graph RAG** — Extraia, conecte e consulte conhecimento através de Grafos de Conhecimento.

## ✨ O que é?

L.O.K.I. é uma CLI interativa que transforma seus documentos em um **Grafo de Conhecimento** (Knowledge Graph). Em vez de buscar por similaridade vetorial, o sistema extrai **entidades** e **relações** dos seus documentos usando LLMs, construindo uma rede semântica navegável e consultável.

## 🏗️ Arquitetura — Graph RAG

```
Documento → Chunking → LLMGraphTransformer → Entidades + Relações → NetworkX DiGraph
                                                                         │
Pergunta → Busca de Entidades → Vizinhança no Grafo → Contexto → LLM → Resposta
                                        │
                                   Pyvis → HTML interativo (navegador)
```

### Stack Tecnológico

| Camada | Tecnologia |
|--------|-----------|
| **CLI / UI** | `rich`, `prompt_toolkit` |
| **Orquestração** | `langchain`, `langchain-experimental` |
| **Extração de Entidades** | `LLMGraphTransformer` |
| **Grafo de Conhecimento** | `networkx` (DiGraph) |
| **Visualização** | `pyvis` (HTML interativo) |
| **LLM Local** | `ollama` (via `ChatOllama`) |
| **LLM Cloud** | OpenAI, Anthropic (opcionais) |

## 📁 Estrutura de Diretórios

```
L.O.K.I/
├── main.py                  # Ponto de entrada
├── requirements.txt         # Dependências
├── .env.example             # Template de configuração
├── config/
│   └── settings.py          # Configurações e factory do LLM
├── graph/
│   ├── __init__.py
│   └── graph_manager.py     # KnowledgeGraph (networkx + pyvis)
├── ingestion/
│   ├── loader.py            # Carregadores de documentos
│   └── ingestion.py         # Pipeline: chunk → LLMGraphTransformer → grafo
├── rag/
│   └── chain.py             # Chain de consulta Graph RAG
├── cli/
│   └── interface.py         # Interface interativa (rich + prompt_toolkit)
└── data/
    ├── knowledge_graph.graphml  # Grafo persistido (gerado automaticamente)
    └── mapa_loki.html           # Visualização HTML (gerada pelo !map)
```

## 🚀 Setup Rápido

```bash
# Ambiente virtual
python -m venv .venv && .venv\Scripts\activate

# Dependências
pip install -r requirements.txt

# Configuração
copy .env.example .env

# Ollama local (se usar provider "ollama")
ollama serve
ollama pull llama3

# Executar
```

## 🎮 Comandos da CLI

| Comando | Descrição |
|---------|-----------|
| `!add <caminho>` | Ingere arquivo/pasta → extrai entidades e relações para o grafo |
| `!map` | Gera e abre visualização interativa do grafo no navegador |
| `!status` | Mostra estatísticas do grafo (entidades, relações, provider) |
| `!help` | Exibe ajuda |
| `sair` | Encerra o L.O.K.I. |
| *texto livre* | Consulta ao Segundo Cérebro via Graph RAG |

## ⚙️ Configuração (.env)

```env
# Provider: ollama | openai | anthropic
LLM_PROVIDER=ollama

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3

# Caminhos do grafo
GRAPH_PERSIST_PATH=./data/knowledge_graph.graphml
GRAPH_VIS_PATH=./data/mapa_loki.html

# Chunking
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

## 🔄 Fluxo de Funcionamento

### Ingestão (`!add`)
1. **Carrega** o documento (PDF, TXT, MD, PY, CSV...)
2. **Fatia** em chunks com `RecursiveCharacterTextSplitter`
3. **Extrai** entidades e relações via `LLMGraphTransformer` (usa o LLM configurado)
4. **Adiciona** as trincas `(sujeito, relação, objeto)` ao `DiGraph` do NetworkX
5. **Persiste** o grafo em disco no formato GraphML

### Consulta (texto livre)
1. **Tokeniza** a pergunta em palavras-chave
2. **Busca** entidades correspondentes no grafo
3. **Expande** a vizinhança (BFS com profundidade configurável)
4. **Formata** as trincas como contexto textual
5. **Invoca** o LLM com o contexto do grafo + pergunta

### Visualização (`!map`)
1. **Carrega** o grafo do NetworkX
2. **Converte** para HTML interativo via `pyvis` com tema escuro
3. **Abre** automaticamente no navegador padrão

## 📜 Licença

MIT
