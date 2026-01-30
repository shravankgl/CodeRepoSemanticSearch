# Code Repository Semantic Search 🔍

A semantic code search agent that indexes GitHub repositories and allows natural language queries to find relevant code, understand architecture, and get AI-assisted code explanations.

## Session 7 Concepts Applied

| Concept | Implementation |
|---------|---------------|
| **RAG (Retrieval Augmented Generation)** | FAISS index for code chunks, semantic search before answering |
| **Memory Management** | `MemoryManager` with vector memory and session state |
| **Perception Layer** | Intent extraction with Claude Haiku |
| **Decision Layer** | Plan generation with FUNCTION_CALL/FINAL_ANSWER format |
| **Action Layer** | MCP tool execution via session |
| **Context Management** | Repository state, indexed chunks, conversation history |

## Architecture

```
┌─────────────────────────────────────────┐
│              User Interface             │
│  "Give me a GitHub repo URL"            │
│  "Ask questions about the repo"         │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│           PERCEPTION LAYER              │
│  - Extract intent (clone/search/etc.)   │
│  - Identify entities (URL, keywords)    │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│             MEMORY LAYER                │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │Session Memory│  │  Vector Memory   │ │
│  │- Repo state  │  │  - FAISS index   │ │
│  │- Chat history│  │  - Code chunks   │ │
│  └──────────────┘  └──────────────────┘ │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│           DECISION LAYER                │
│  - Generate plan using Claude Haiku     │
│  - FUNCTION_CALL or FINAL_ANSWER        │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│            ACTION LAYER                 │
│  Execute MCP tools via Agent Session    │
│  - clone_repository                     │
│  - index_repository                     │
│  - search_code                          │
│  - get_file_content                     │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│         CODE PROCESSING LAYER           │
│  ┌──────────────────┐  ┌──────────────┐ │
│  │Language Detector │  │  AST Parser  │ │
│  └────────┬─────────┘  └──────┬───────┘ │
│           │                   │         │
│           ▼                   ▼         │
│   [Repo Language]      [Code Chunks]    │
└─────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

Using [uv](https://github.com/astral-sh/uv) for fast package management:

```bash
# Initialize project and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
OLLAMA_EMBED_URL=http://localhost:11434/api/embeddings
EMBED_MODEL=nomic-embed-text
```

### 3. Start Ollama (for embeddings)

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Start Ollama server (if not running)
ollama serve
```

## Usage

```bash
# Run the agent with uv
uv run agent.py
```

### Example Workflow

```
🔍 Welcome
Commands:
  • Paste a GitHub URL to clone and index a repository
  • Ask questions about the code
  • Type 'status' to check repository status
  • Type 'exit' or 'quit' to exit

You: https://github.com/pallets/flask

🧠 Analyzing input...
   Intent: clone_repo | Tool hint: clone_repository
🤔 Planning (step 1/10)...
🔧 Calling tool: clone_repository
✓ Successfully cloned flask to ./repos/flask

🔧 Calling tool: index_repository
✓ Indexed 1,234 code chunks from 150 files

You: How does Flask handle routing?

🧠 Analyzing input...
   Intent: search_code | Tool hint: search_code
🤔 Planning (step 1/10)...
🔧 Calling tool: search_code
✓ Found 5 matching code chunks

📝 Synthesizing answer from search results...

┌─────────────────────────────────────────┐
│ Flask handles routing through the       │
│ `@app.route()` decorator which          │
│ internally uses the `Rule` class...     │
│                                         │
│ [1] src/flask/app.py (lines 45-89)     │
│ [2] src/flask/helpers.py (lines 12-45) │
└─────────────────────────────────────────┘
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `clone_repository` | Clone a GitHub repository |
| `index_repository` | Index code for semantic search |
| `search_code` | Natural language code search |
| `get_file_content` | Read file contents |
| `list_files` | List files matching pattern |
| `get_repo_status` | Get repository info |
| `load_repository` | Load previously indexed repo |

## File Structure

```
CodeRepoSemanticSearch/
├── agent.py          # Main entry point
├── config.py         # Configuration settings
├── models.py         # Pydantic models
├── perception.py     # Intent extraction (Claude)
├── decision.py       # Plan generation (Claude)
├── action.py         # Tool execution
├── memory.py         # FAISS + session memory
├── code_chunker.py   # AST-based code parsing
├── repo_tools.py     # MCP server with tools
├── requirements.txt  # Dependencies
├── repos/            # Cloned repositories
└── indexes/          # FAISS indexes
```

## Technologies

- **LLM**: Claude Haiku (claude-haiku-4-5-20251001)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Search**: FAISS
- **Code Parsing**: tree-sitter
- **Protocol**: MCP (Model Context Protocol)
- **UI**: Rich (terminal)

## Intelligent Language Detection

The agent automatically detects the primary language of a repository by analyzing file extensions and loads the appropriate AST parser.

### AST Support by Language

| Language | Parser | Status |
|----------|--------|--------|
| Python | `tree-sitter-python` | ✅ Included |
| JavaScript | `tree-sitter-javascript` | 📦 Optional |
| TypeScript | `tree-sitter-typescript` | 📦 Optional |
| Java | `tree-sitter-java` | 📦 Optional |
| Rust | `tree-sitter-rust` | 📦 Optional |
| Go | `tree-sitter-go` | 📦 Optional |
| C/C++ | `tree-sitter-c/cpp` | 📦 Optional |
| Ruby | `tree-sitter-ruby` | 📦 Optional |

**To enable AST support for additional languages:**
```bash
pip install tree-sitter-javascript tree-sitter-java tree-sitter-rust
```

Languages without an installed parser will automatically fall back to text-based chunking.

### All Supported File Types

| Extension | Language | Chunking |
|-----------|----------|----------|
| `.py` | Python | AST (functions, classes, methods) |
| `.js`, `.jsx` | JavaScript | AST or Text fallback |
| `.ts`, `.tsx` | TypeScript | AST or Text fallback |
| `.java` | Java | AST or Text fallback |
| `.rs` | Rust | AST or Text fallback |
| `.go` | Go | AST or Text fallback |
| `.c`, `.h`, `.cpp`, `.hpp` | C/C++ | AST or Text fallback |
| `.rb` | Ruby | AST or Text fallback |
| `.swift`, `.kt`, `.scala`, `.cs` | Others | Text-based |
| `.md`, `.json`, `.yaml`, `.toml` | Config | Text-based |

## License

MIT
