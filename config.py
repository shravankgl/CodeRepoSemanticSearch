"""
Configuration settings for Code Repository Semantic Search
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
REPOS_DIR = BASE_DIR / "repos"
INDEX_DIR = BASE_DIR / "indexes"

# Create directories if they don't exist
REPOS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Claude Model
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Ollama Embedding Settings
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIMENSION = 768  # nomic-embed-text dimension

# Chunking Settings
CHUNK_SIZE = 256  # words per chunk
CHUNK_OVERLAP = 40  # overlap between chunks

# Agent Settings
MAX_AGENT_STEPS = 10
REQUEST_TIMEOUT = 60

# Rate Limiting (for Claude Tier 1)
MAX_REQUESTS_PER_MINUTE = 5

# Supported file extensions for indexing
SUPPORTED_EXTENSIONS = {
    # AST-based chunking (Python only currently)
    ".py": "python",
    # Text-based chunking for other languages
    ".js": "javascript", 
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

# Files/directories to ignore during indexing
IGNORE_PATTERNS = [
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    ".env",
    ".idea",
    ".vscode",
]
