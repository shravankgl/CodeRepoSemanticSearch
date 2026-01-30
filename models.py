"""
Pydantic models for Code Repository Semantic Search
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# ============== Perception Models ==============

class PerceptionResult(BaseModel):
    """Result of analyzing user input."""
    user_input: str
    intent: Optional[str] = None  # clone_repo, search_code, explain_code, list_files, get_status
    entities: List[str] = Field(default_factory=list)
    tool_hint: Optional[str] = None
    repo_url: Optional[str] = None


# ============== Memory Models ==============

class MemoryItem(BaseModel):
    """Item stored in semantic memory."""
    text: str
    type: Literal["tool_output", "fact", "query", "code_chunk", "system"] = "fact"
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class RepositoryState(BaseModel):
    """State of the currently indexed repository."""
    repo_url: Optional[str] = None
    repo_name: Optional[str] = None
    local_path: Optional[str] = None
    is_cloned: bool = False
    is_indexed: bool = False
    file_count: int = 0
    chunk_count: int = 0
    last_indexed: Optional[str] = None


# ============== Code Chunk Models ==============

class CodeChunk(BaseModel):
    """Represents a chunk of code (function, class, or text block)."""
    content: str
    file_path: str
    chunk_type: Literal["function", "class", "method", "text"] = "text"
    name: Optional[str] = None  # function/class name
    start_line: int = 0
    end_line: int = 0
    language: str = "python"
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from code search."""
    chunk: CodeChunk
    score: float
    rank: int


# ============== Tool Models ==============

class CloneRepositoryInput(BaseModel):
    """Input for clone_repository tool."""
    repo_url: str


class CloneRepositoryOutput(BaseModel):
    """Output from clone_repository tool."""
    success: bool
    local_path: str
    message: str


class IndexRepositoryInput(BaseModel):
    """Input for index_repository tool."""
    repo_path: Optional[str] = None  # If None, use current repo


class IndexRepositoryOutput(BaseModel):
    """Output from index_repository tool."""
    success: bool
    file_count: int
    chunk_count: int
    message: str


class SearchCodeInput(BaseModel):
    """Input for search_code tool."""
    query: str
    top_k: int = 5


class SearchCodeOutput(BaseModel):
    """Output from search_code tool."""
    results: List[Dict[str, Any]]
    total_found: int


class GetFileContentInput(BaseModel):
    """Input for get_file_content tool."""
    file_path: str


class GetFileContentOutput(BaseModel):
    """Output from get_file_content tool."""
    content: str
    file_path: str
    line_count: int


class ExplainCodeInput(BaseModel):
    """Input for explain_code tool."""
    code_snippet: str
    context: Optional[str] = None


class ExplainCodeOutput(BaseModel):
    """Output from explain_code tool."""
    explanation: str


class ListFilesInput(BaseModel):
    """Input for list_files tool."""
    pattern: str = "*.py"


class ListFilesOutput(BaseModel):
    """Output from list_files tool."""
    files: List[str]
    total_count: int


class GetRepoStatusOutput(BaseModel):
    """Output from get_repo_status tool."""
    repo_url: Optional[str]
    repo_name: Optional[str]
    is_cloned: bool
    is_indexed: bool
    file_count: int
    chunk_count: int


# ============== Action Models ==============

class ToolCallResult(BaseModel):
    """Result of executing a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None
