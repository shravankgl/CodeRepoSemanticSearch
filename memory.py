"""
Memory Layer - State management, context, and vector memory

Implements:
- VectorMemory: FAISS-based semantic memory with nomic embeddings
- SessionMemory: Conversation history and repository state
- MemoryManager: Unified interface for all memory operations
"""

import sys
import json
import numpy as np
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[memory] Warning: faiss not available", file=sys.stderr)

from pydantic import BaseModel, Field
from models import MemoryItem, RepositoryState, CodeChunk
from config import (
    OLLAMA_EMBED_URL, EMBED_MODEL, EMBED_DIMENSION,
    INDEX_DIR
)


class Message(BaseModel):
    """Single conversation message."""
    role: str  # "user", "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class VectorMemory:
    """
    FAISS-based semantic memory for code chunks.
    Uses nomic-embed-text via Ollama for embeddings.
    """
    
    def __init__(self, repo_name: str = "default"):
        self.repo_name = repo_name
        self.index = None
        self.chunks: List[CodeChunk] = []
        self.embeddings: List[np.ndarray] = []
        
        # Paths for persistence
        self.index_path = INDEX_DIR / repo_name
        self.index_path.mkdir(parents=True, exist_ok=True)
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama nomic-embed-text model."""
        try:
            response = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = np.array(response.json()["embedding"], dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"[memory] Embedding error: {e}", file=sys.stderr)
            # Return zero vector on error
            return np.zeros(EMBED_DIMENSION, dtype=np.float32)
    
    def add_chunk(self, chunk: CodeChunk):
        """Add a code chunk to the vector index."""
        if not FAISS_AVAILABLE:
            print("[memory] FAISS not available, skipping indexing", file=sys.stderr)
            return
            
        # Create embedding from chunk content
        text_for_embedding = f"{chunk.name or ''}\n{chunk.docstring or ''}\n{chunk.content}"
        embedding = self._get_embedding(text_for_embedding[:2000])  # Limit text length
        
        self.embeddings.append(embedding)
        self.chunks.append(chunk)
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBED_DIMENSION)  # Inner product for similarity
        
        # Normalize for cosine similarity
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.index.add(np.stack([norm_embedding]))
    
    def add_chunks(self, chunks: List[CodeChunk], show_progress: bool = True):
        """Add multiple chunks with optional progress display."""
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                print(f"[memory] Indexing chunks: {i+1}/{total}", file=sys.stderr)
            self.add_chunk(chunk)
        print(f"[memory] Indexed {total} chunks", file=sys.stderr)
    
    def search(self, query: str, top_k: int = 5) -> List[tuple[CodeChunk, float]]:
        """
        Search for similar code chunks.
        
        Returns:
            List of (CodeChunk, score) tuples
        """
        if not FAISS_AVAILABLE or self.index is None or len(self.chunks) == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_array = query_embedding.reshape(1, -1)
        
        # Search
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_array, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self):
        """Save index and metadata to disk."""
        if not FAISS_AVAILABLE or self.index is None:
            return
        
        # Save FAISS index
        index_file = self.index_path / "index.bin"
        faiss.write_index(self.index, str(index_file))
        
        # Save chunk metadata
        metadata = []
        for chunk in self.chunks:
            metadata.append(chunk.model_dump())
        
        metadata_file = self.index_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
        
        print(f"[memory] Saved index to {self.index_path}", file=sys.stderr)
    
    def load(self) -> bool:
        """Load index and metadata from disk."""
        if not FAISS_AVAILABLE:
            return False
            
        index_file = self.index_path / "index.bin"
        metadata_file = self.index_path / "metadata.json"
        
        if not index_file.exists() or not metadata_file.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_file))
            metadata = json.loads(metadata_file.read_text())
            self.chunks = [CodeChunk(**m) for m in metadata]
            print(f"[memory] Loaded {len(self.chunks)} chunks from {self.index_path}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[memory] Failed to load index: {e}", file=sys.stderr)
            return False
    
    def clear(self):
        """Clear the index."""
        self.index = None
        self.chunks = []
        self.embeddings = []


class SessionMemory:
    """
    Session memory for conversation history and repository state.
    """
    
    def __init__(self):
        self.conversation_history: List[Message] = []
        self.repo_state = RepositoryState()
        self.tool_outputs: List[Dict[str, Any]] = []
        self.session_id: str = f"session-{int(datetime.now().timestamp())}"
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append(
            Message(role=role, content=content)
        )
    
    def add_tool_output(self, tool_name: str, arguments: Dict, result: Any):
        """Store a tool output for context."""
        self.tool_outputs.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_context(self, n: int = 5) -> List[Message]:
        """Get last n messages."""
        return self.conversation_history[-n:]
    
    def get_context_string(self) -> str:
        """Get conversation as formatted string for LLM context."""
        if not self.conversation_history:
            return "No previous conversation."
        
        lines = []
        for msg in self.conversation_history[-5:]:
            content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"{msg.role}: {content_preview}")
        return "\n".join(lines)
    
    def get_recent_tool_outputs(self, n: int = 3) -> str:
        """Get recent tool outputs as context."""
        if not self.tool_outputs:
            return "No previous tool outputs."
        
        lines = []
        for output in self.tool_outputs[-n:]:
            result_preview = str(output["result"])[:300]
            lines.append(f"- {output['tool_name']}: {result_preview}")
        return "\n".join(lines)
    
    def set_repo_state(self, **kwargs):
        """Update repository state."""
        for key, value in kwargs.items():
            if hasattr(self.repo_state, key):
                setattr(self.repo_state, key, value)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.tool_outputs = []


class MemoryManager:
    """
    Unified memory manager combining vector and session memory.
    """
    
    def __init__(self):
        self.session = SessionMemory()
        self.vector: Optional[VectorMemory] = None
    
    def init_vector_memory(self, repo_name: str):
        """Initialize vector memory for a repository."""
        self.vector = VectorMemory(repo_name)
        # Try to load existing index
        self.vector.load()
    
    def add_message(self, role: str, content: str):
        """Add message to session."""
        self.session.add_message(role, content)
    
    def add_tool_output(self, tool_name: str, arguments: Dict, result: Any):
        """Add tool output to session."""
        self.session.add_tool_output(tool_name, arguments, result)
    
    def search_code(self, query: str, top_k: int = 5) -> List[tuple[CodeChunk, float]]:
        """Search code in vector memory."""
        if self.vector is None:
            return []
        return self.vector.search(query, top_k)
    
    def get_repo_state(self) -> RepositoryState:
        """Get current repository state."""
        return self.session.repo_state
    
    def set_repo_state(self, **kwargs):
        """Update repository state."""
        self.session.set_repo_state(**kwargs)
    
    def get_context_for_llm(self) -> str:
        """Get combined context for LLM prompts."""
        parts = []
        
        # Repository state
        repo = self.session.repo_state
        if repo.is_indexed:
            parts.append(f"Repository: {repo.repo_name} ({repo.chunk_count} code chunks indexed)")
        elif repo.is_cloned:
            parts.append(f"Repository: {repo.repo_name} (cloned, not indexed)")
        else:
            parts.append("No repository loaded")
        
        # Recent conversation
        conv = self.session.get_context_string()
        if conv != "No previous conversation.":
            parts.append(f"\nRecent conversation:\n{conv}")
        
        # Recent tool outputs
        tools = self.session.get_recent_tool_outputs()
        if tools != "No previous tool outputs.":
            parts.append(f"\nRecent tool outputs:\n{tools}")
        
        return "\n".join(parts)
    
    def save_vector_index(self):
        """Save vector index to disk."""
        if self.vector:
            self.vector.save()
    
    def clear(self):
        """Clear all memory."""
        self.session.clear_history()
        if self.vector:
            self.vector.clear()
