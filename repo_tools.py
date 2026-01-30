"""
Repository Tools - MCP server with code repository tools

Provides tools for:
- Cloning GitHub repositories
- Indexing code with semantic embeddings
- Searching code with natural language
- Getting file contents
- Explaining code
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from config import (
    REPOS_DIR, INDEX_DIR, SUPPORTED_EXTENSIONS, IGNORE_PATTERNS,
    BASE_DIR
)
from models import CodeChunk
from code_chunker import get_chunker
from memory import VectorMemory

# Initialize MCP server
mcp = FastMCP("CodeRepoSearch")

# Global state for the current repository
_current_repo: Optional[str] = None
_vector_memory: Optional[VectorMemory] = None


def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication."""
    sys.stderr.write(f"[{level}] {message}\n")
    sys.stderr.flush()


def get_repo_name_from_url(url: str) -> str:
    """Extract repository name from GitHub URL."""
    # Handle various URL formats
    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    return url.split('/')[-1]


def should_ignore(path: Path) -> bool:
    """Check if a path should be ignored during indexing."""
    path_str = str(path)
    for pattern in IGNORE_PATTERNS:
        if pattern in path_str:
            return True
    return False


@mcp.tool()
def clone_repository(repo_url: str) -> str:
    """
    Clone a GitHub repository to the local repos directory.
    
    Args:
        repo_url: Full GitHub repository URL (e.g., https://github.com/user/repo)
    
    Returns:
        Status message with the local path
    """
    global _current_repo, _vector_memory
    
    mcp_log("CLONE", f"Cloning repository: {repo_url}")
    
    try:
        # Import git here to avoid issues if not installed
        import git
        
        repo_name = get_repo_name_from_url(repo_url)
        local_path = REPOS_DIR / repo_name
        
        # Remove if already exists
        if local_path.exists():
            mcp_log("INFO", f"Removing existing repository at {local_path}")
            shutil.rmtree(local_path)
        
        # Clone the repository
        mcp_log("INFO", f"Cloning to {local_path}...")
        git.Repo.clone_from(repo_url, local_path, depth=1)  # Shallow clone for speed
        
        # Update global state
        _current_repo = repo_name
        _vector_memory = VectorMemory(repo_name)
        
        # Count files
        file_count = sum(1 for f in local_path.rglob("*") if f.is_file() and not should_ignore(f))
        
        mcp_log("SUCCESS", f"Cloned {repo_name} with {file_count} files")
        
        return json.dumps({
            "success": True,
            "repo_name": repo_name,
            "local_path": str(local_path),
            "file_count": file_count,
            "message": f"Successfully cloned {repo_name} to {local_path}"
        })
        
    except Exception as e:
        mcp_log("ERROR", f"Clone failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to clone repository: {e}"
        })


@mcp.tool()
def index_repository(repo_path: Optional[str] = None) -> str:
    """
    Index the repository code files for semantic search.
    Creates embeddings for all code chunks and stores in FAISS index.
    
    Args:
        repo_path: Optional path to repository. If not provided, uses the last cloned repo.
    
    Returns:
        Status message with indexing statistics
    """
    global _current_repo, _vector_memory
    
    mcp_log("INDEX", "Starting repository indexing...")
    
    try:
        # Determine repo path
        if repo_path:
            path = Path(repo_path)
            repo_name = path.name
        elif _current_repo:
            path = REPOS_DIR / _current_repo
            repo_name = _current_repo
        else:
            return json.dumps({
                "success": False,
                "error": "No repository specified and no repo currently loaded",
                "message": "Please clone a repository first or specify a path"
            })
        
        if not path.exists():
            return json.dumps({
                "success": False,
                "error": f"Repository path does not exist: {path}",
                "message": "Clone the repository first"
            })
        
        # Initialize vector memory
        _vector_memory = VectorMemory(repo_name)
        _current_repo = repo_name
        
        # Get chunker
        chunker = get_chunker()
        
        # Find and process all code files
        all_chunks: List[CodeChunk] = []
        file_count = 0
        
        for ext, language in SUPPORTED_EXTENSIONS.items():
            for file_path in path.rglob(f"*{ext}"):
                if should_ignore(file_path):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Skip very small or very large files
                    if len(content) < 50 or len(content) > 100000:
                        continue
                    
                    # Get relative path for storage
                    rel_path = str(file_path.relative_to(path))
                    
                    # Chunk the file
                    chunks = chunker.chunk_file(rel_path, content, language)
                    all_chunks.extend(chunks)
                    file_count += 1
                    
                    if file_count % 20 == 0:
                        mcp_log("PROGRESS", f"Processed {file_count} files, {len(all_chunks)} chunks")
                        
                except Exception as e:
                    mcp_log("WARN", f"Failed to process {file_path}: {e}")
        
        mcp_log("INFO", f"Found {len(all_chunks)} chunks from {file_count} files")
        
        # Add chunks to vector memory
        if all_chunks:
            mcp_log("INFO", "Creating embeddings and building index...")
            _vector_memory.add_chunks(all_chunks, show_progress=True)
            _vector_memory.save()
        
        return json.dumps({
            "success": True,
            "repo_name": repo_name,
            "file_count": file_count,
            "chunk_count": len(all_chunks),
            "message": f"Indexed {len(all_chunks)} code chunks from {file_count} files"
        })
        
    except Exception as e:
        mcp_log("ERROR", f"Indexing failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to index repository: {e}"
        })


@mcp.tool()
def search_code(query: str, top_k: int = 5) -> str:
    """
    Search for relevant code using natural language.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 5)
    
    Returns:
        Matching code chunks with file paths and scores
    """
    global _vector_memory
    
    mcp_log("SEARCH", f"Searching for: {query}")
    
    try:
        if _vector_memory is None:
            # Try to load from disk if repo was previously indexed
            if _current_repo:
                _vector_memory = VectorMemory(_current_repo)
                if not _vector_memory.load():
                    return json.dumps({
                        "success": False,
                        "error": "No index loaded",
                        "message": "Please index a repository first"
                    })
            else:
                return json.dumps({
                    "success": False,
                    "error": "No repository indexed",
                    "message": "Please clone and index a repository first"
                })
        
        # Search
        results = _vector_memory.search(query, top_k)
        
        if not results:
            return json.dumps({
                "success": True,
                "results": [],
                "total_found": 0,
                "message": "No matching code found"
            })
        
        # Format results
        formatted_results = []
        for i, (chunk, score) in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "score": round(score, 4),
                "file_path": chunk.file_path,
                "name": chunk.name,
                "type": chunk.chunk_type,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                "docstring": chunk.docstring
            })
        
        mcp_log("SUCCESS", f"Found {len(results)} results")
        
        return json.dumps({
            "success": True,
            "results": formatted_results,
            "total_found": len(results),
            "message": f"Found {len(results)} matching code chunks"
        })
        
    except Exception as e:
        mcp_log("ERROR", f"Search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Search failed: {e}"
        })


@mcp.tool()
def get_file_content(file_path: str) -> str:
    """
    Get the full content of a file in the repository.
    
    Args:
        file_path: Relative path to the file within the repository
    
    Returns:
        File content with line numbers
    """
    global _current_repo
    
    mcp_log("FILE", f"Getting content: {file_path}")
    
    try:
        if not _current_repo:
            return json.dumps({
                "success": False,
                "error": "No repository loaded",
                "message": "Please clone a repository first"
            })
        
        full_path = REPOS_DIR / _current_repo / file_path
        
        if not full_path.exists():
            return json.dumps({
                "success": False,
                "error": f"File not found: {file_path}",
                "message": "The file does not exist in the repository"
            })
        
        content = full_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        # Add line numbers
        numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        numbered_content = '\n'.join(numbered_lines)
        
        return json.dumps({
            "success": True,
            "file_path": file_path,
            "content": numbered_content,
            "line_count": len(lines),
            "message": f"Retrieved {len(lines)} lines from {file_path}"
        })
        
    except Exception as e:
        mcp_log("ERROR", f"Failed to read file: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to read file: {e}"
        })


@mcp.tool()
def list_files(pattern: str = "*.py") -> str:
    """
    List files in the repository matching a pattern.
    
    Args:
        pattern: Glob pattern to match files (default: *.py)
    
    Returns:
        List of matching file paths
    """
    global _current_repo
    
    mcp_log("LIST", f"Listing files with pattern: {pattern}")
    
    try:
        if not _current_repo:
            return json.dumps({
                "success": False,
                "error": "No repository loaded",
                "message": "Please clone a repository first"
            })
        
        repo_path = REPOS_DIR / _current_repo
        
        # Find matching files
        files = []
        for file_path in repo_path.rglob(pattern):
            if file_path.is_file() and not should_ignore(file_path):
                rel_path = str(file_path.relative_to(repo_path))
                files.append(rel_path)
        
        files.sort()
        
        return json.dumps({
            "success": True,
            "files": files[:100],  # Limit to 100 files
            "total_count": len(files),
            "pattern": pattern,
            "message": f"Found {len(files)} files matching '{pattern}'"
        })
        
    except Exception as e:
        mcp_log("ERROR", f"Failed to list files: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to list files: {e}"
        })


@mcp.tool()
def get_repo_status() -> str:
    """
    Get the current status of the loaded repository.
    
    Returns:
        Repository information including index status
    """
    global _current_repo, _vector_memory
    
    mcp_log("STATUS", "Getting repository status")
    
    try:
        if not _current_repo:
            return json.dumps({
                "success": True,
                "is_loaded": False,
                "message": "No repository currently loaded. Use clone_repository to load one."
            })
        
        repo_path = REPOS_DIR / _current_repo
        
        # Count files
        file_count = sum(1 for f in repo_path.rglob("*") if f.is_file() and not should_ignore(f))
        
        # Check index status
        is_indexed = False
        chunk_count = 0
        if _vector_memory and _vector_memory.chunks:
            is_indexed = True
            chunk_count = len(_vector_memory.chunks)
        else:
            # Check if index exists on disk
            index_path = INDEX_DIR / _current_repo / "index.bin"
            if index_path.exists():
                is_indexed = True
        
        return json.dumps({
            "success": True,
            "is_loaded": True,
            "repo_name": _current_repo,
            "local_path": str(repo_path),
            "file_count": file_count,
            "is_indexed": is_indexed,
            "chunk_count": chunk_count,
            "message": f"Repository '{_current_repo}' loaded with {file_count} files"
        })
        
    except Exception as e:
        mcp_log("ERROR", f"Status check failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to get status: {e}"
        })


@mcp.tool() 
def load_repository(repo_name: str) -> str:
    """
    Load a previously cloned and indexed repository.
    
    Args:
        repo_name: Name of the repository to load
    
    Returns:
        Status message
    """
    global _current_repo, _vector_memory
    
    mcp_log("LOAD", f"Loading repository: {repo_name}")
    
    try:
        repo_path = REPOS_DIR / repo_name
        
        if not repo_path.exists():
            return json.dumps({
                "success": False,
                "error": f"Repository not found: {repo_name}",
                "message": f"Repository '{repo_name}' does not exist. Clone it first."
            })
        
        _current_repo = repo_name
        _vector_memory = VectorMemory(repo_name)
        
        # Try to load existing index
        if _vector_memory.load():
            return json.dumps({
                "success": True,
                "repo_name": repo_name,
                "chunk_count": len(_vector_memory.chunks),
                "message": f"Loaded repository '{repo_name}' with {len(_vector_memory.chunks)} indexed chunks"
            })
        else:
            return json.dumps({
                "success": True,
                "repo_name": repo_name,
                "chunk_count": 0,
                "message": f"Loaded repository '{repo_name}' (not indexed yet)"
            })
        
    except Exception as e:
        mcp_log("ERROR", f"Load failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to load repository: {e}"
        })


if __name__ == "__main__":
    mcp_log("INFO", "Starting Code Repository Search MCP server...")
    mcp.run(transport="stdio")
