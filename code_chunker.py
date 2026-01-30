"""
Code Chunker - Multi-language AST-based code chunking

Features:
- Intelligent language detection for repositories
- AST parsing for Python, JavaScript, TypeScript, Java, Rust, Go
- Automatic fallback to text-based chunking
- Dynamic parser loading
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter
from models import CodeChunk
from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

# Language to tree-sitter module mapping
LANGUAGE_PARSERS = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "rust": "tree_sitter_rust",
    "go": "tree_sitter_go",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
}

# Node types for extracting code elements per language
LANGUAGE_NODE_TYPES = {
    "python": {
        "function": ["function_definition"],
        "class": ["class_definition"],
        "method": ["function_definition"],
    },
    "javascript": {
        "function": ["function_declaration", "arrow_function", "function_expression"],
        "class": ["class_declaration"],
        "method": ["method_definition"],
    },
    "typescript": {
        "function": ["function_declaration", "arrow_function", "function_expression"],
        "class": ["class_declaration", "interface_declaration"],
        "method": ["method_definition", "method_signature"],
    },
    "java": {
        "function": ["method_declaration", "constructor_declaration"],
        "class": ["class_declaration", "interface_declaration"],
        "method": ["method_declaration"],
    },
    "rust": {
        "function": ["function_item"],
        "class": ["struct_item", "impl_item", "trait_item"],
        "method": ["function_item"],
    },
    "go": {
        "function": ["function_declaration", "method_declaration"],
        "class": ["type_declaration"],
        "method": ["method_declaration"],
    },
    "c": {
        "function": ["function_definition"],
        "class": ["struct_specifier"],
        "method": [],
    },
    "cpp": {
        "function": ["function_definition"],
        "class": ["class_specifier", "struct_specifier"],
        "method": ["function_definition"],
    },
    "ruby": {
        "function": ["method"],
        "class": ["class", "module"],
        "method": ["method"],
    },
}


def log(msg: str):
    """Log to stderr."""
    sys.stderr.write(f"[code_chunker] {msg}\n")
    sys.stderr.flush()


class LanguageDetector:
    """
    Intelligent language detection for repositories.
    Analyzes file extensions to determine the primary language.
    """
    
    @staticmethod
    def detect_repo_language(repo_path: Path) -> str:
        """
        Detect the primary programming language of a repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Primary language name (e.g., "python", "javascript", "java")
        """
        extension_counts = Counter()
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    lang = SUPPORTED_EXTENSIONS[ext]
                    extension_counts[lang] += 1
        
        if not extension_counts:
            return "python"  # Default
        
        # Get the most common language
        primary_lang, count = extension_counts.most_common(1)[0]
        total_files = sum(extension_counts.values())
        
        log(f"Language detection: {primary_lang} ({count}/{total_files} files, "
            f"{count/total_files*100:.1f}%)")
        log(f"  All languages: {dict(extension_counts.most_common(5))}")
        
        return primary_lang
    
    @staticmethod
    def get_file_language(file_path: str) -> str:
        """Get the language of a specific file based on extension."""
        ext = Path(file_path).suffix.lower()
        return SUPPORTED_EXTENSIONS.get(ext, "text")


class MultiLanguageChunker:
    """
    Multi-language code chunker with AST parsing support.
    Dynamically loads tree-sitter parsers for detected languages.
    """
    
    def __init__(self):
        self.parsers: Dict[str, any] = {}
        self.available_languages: List[str] = []
        self._detect_available_parsers()
    
    def _detect_available_parsers(self):
        """Detect which tree-sitter language parsers are installed."""
        try:
            from tree_sitter import Language, Parser
            self._tree_sitter_available = True
        except ImportError:
            log("tree-sitter not installed, using text-based chunking only")
            self._tree_sitter_available = False
            return
        
        for lang, module_name in LANGUAGE_PARSERS.items():
            try:
                module = __import__(module_name)
                self.available_languages.append(lang)
            except ImportError:
                pass
        
        if self.available_languages:
            log(f"AST parsers available for: {', '.join(self.available_languages)}")
        else:
            log("No tree-sitter language parsers found, using text-based chunking")
    
    def _get_parser(self, language: str):
        """Get or create a parser for the specified language."""
        if not self._tree_sitter_available:
            return None
        
        if language not in self.available_languages:
            return None
        
        if language not in self.parsers:
            try:
                from tree_sitter import Language, Parser
                module_name = LANGUAGE_PARSERS[language]
                module = __import__(module_name)
                
                parser = Parser()
                
                # Handle TypeScript specially (has separate tsx and typescript)
                if language == "typescript":
                    try:
                        # Try typescript first
                        parser.language = Language(module.language_typescript())
                    except AttributeError:
                        parser.language = Language(module.language())
                else:
                    parser.language = Language(module.language())
                
                self.parsers[language] = parser
                log(f"Loaded AST parser for {language}")
                
            except Exception as e:
                log(f"Failed to load parser for {language}: {e}")
                return None
        
        return self.parsers.get(language)
    
    def chunk_file(self, file_path: str, content: str, language: str = None) -> List[CodeChunk]:
        """
        Chunk a file's content into semantic units.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language (auto-detected if None)
            
        Returns:
            List of CodeChunk objects
        """
        if language is None:
            language = LanguageDetector.get_file_language(file_path)
        
        # Try AST parsing first
        parser = self._get_parser(language)
        if parser is not None:
            chunks = self._chunk_with_ast(file_path, content, language, parser)
            if chunks:
                return chunks
        
        # Fall back to text-based chunking
        return self._chunk_text(file_path, content, language)
    
    def _chunk_with_ast(self, file_path: str, content: str, language: str, parser) -> List[CodeChunk]:
        """
        Chunk code using AST parsing.
        """
        chunks = []
        
        try:
            tree = parser.parse(bytes(content, "utf8"))
            root = tree.root_node
            
            node_types = LANGUAGE_NODE_TYPES.get(language, {})
            function_types = node_types.get("function", [])
            class_types = node_types.get("class", [])
            method_types = node_types.get("method", [])
            
            # Recursively extract code elements
            self._extract_nodes(
                root, content, file_path, language,
                function_types, class_types, method_types,
                chunks
            )
            
        except Exception as e:
            log(f"AST parsing failed for {file_path}: {e}")
            return []
        
        return chunks
    
    def _extract_nodes(
        self, node, content: str, file_path: str, language: str,
        function_types: List[str], class_types: List[str], method_types: List[str],
        chunks: List[CodeChunk], class_name: str = None
    ):
        """Recursively extract code nodes from AST."""
        
        for child in node.children:
            if child.type in function_types:
                chunk = self._create_chunk(
                    child, content, file_path, language,
                    "function" if class_name is None else "method",
                    class_name
                )
                if chunk:
                    chunks.append(chunk)
                    
            elif child.type in class_types:
                # Extract class/struct
                chunk = self._create_chunk(
                    child, content, file_path, language, "class"
                )
                if chunk:
                    chunks.append(chunk)
                
                # Extract methods within the class
                name = self._get_node_name(child, language)
                self._extract_nodes(
                    child, content, file_path, language,
                    method_types, [], [],
                    chunks, class_name=name
                )
            else:
                # Recursively search in other nodes
                self._extract_nodes(
                    child, content, file_path, language,
                    function_types, class_types, method_types,
                    chunks, class_name
                )
    
    def _create_chunk(
        self, node, content: str, file_path: str, language: str,
        chunk_type: str, class_name: str = None
    ) -> Optional[CodeChunk]:
        """Create a CodeChunk from an AST node."""
        try:
            code = content[node.start_byte:node.end_byte]
            name = self._get_node_name(node, language)
            docstring = self._get_docstring(node, content, language)
            
            if class_name and chunk_type == "method":
                name = f"{class_name}.{name}"
            
            # Truncate very large chunks
            if len(code.split('\n')) > 100:
                lines = code.split('\n')
                code = '\n'.join(lines[:50]) + f'\n    # ... ({len(lines)-50} more lines)'
            
            return CodeChunk(
                content=code,
                file_path=file_path,
                chunk_type=chunk_type,
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
                docstring=docstring,
                metadata={"type": chunk_type, "class": class_name} if class_name else {"type": chunk_type}
            )
        except Exception as e:
            log(f"Failed to create chunk: {e}")
            return None
    
    def _get_node_name(self, node, language: str) -> str:
        """Get the name of a node (function/class name)."""
        name_node_types = ["identifier", "name", "type_identifier", "constant"]
        
        for child in node.children:
            if child.type in name_node_types:
                return child.text.decode("utf8")
            # For some languages, name might be nested
            if child.type in ["declarator", "function_declarator"]:
                return self._get_node_name(child, language)
        
        return "unknown"
    
    def _get_docstring(self, node, content: str, language: str) -> Optional[str]:
        """Extract docstring/comment from a node."""
        try:
            # Look for string or comment at the start of the function body
            for child in node.children:
                if child.type in ["block", "body", "compound_statement", "block_statement"]:
                    for block_child in child.children:
                        if block_child.type in ["expression_statement", "string", "comment"]:
                            text = content[block_child.start_byte:block_child.end_byte]
                            # Clean up
                            text = text.strip('"""').strip("'''").strip("/*").strip("*/").strip()
                            return text[:500] if len(text) > 500 else text
                    break
                elif child.type == "comment":
                    text = content[child.start_byte:child.end_byte]
                    return text[:500].strip("//").strip("/*").strip("*/").strip()
        except Exception:
            pass
        return None
    
    def _chunk_text(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """
        Text-based chunking for unsupported languages or fallback.
        """
        chunks = []
        lines = content.split('\n')
        
        current_chunk_lines = []
        current_start_line = 1
        current_word_count = 0
        
        for i, line in enumerate(lines, 1):
            word_count = len(line.split())
            current_chunk_lines.append(line)
            current_word_count += word_count
            
            is_boundary = (
                line.strip() == '' and current_word_count > CHUNK_SIZE // 2 or
                current_word_count >= CHUNK_SIZE
            )
            
            if is_boundary and current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines).strip()
                if chunk_content:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        chunk_type="text",
                        name=None,
                        start_line=current_start_line,
                        end_line=i,
                        language=language,
                        docstring=None,
                        metadata={"type": "text"}
                    ))
                
                overlap_lines = current_chunk_lines[-CHUNK_OVERLAP // 10:] if CHUNK_OVERLAP > 0 else []
                current_chunk_lines = overlap_lines
                current_start_line = max(1, i - len(overlap_lines) + 1)
                current_word_count = sum(len(l.split()) for l in current_chunk_lines)
        
        # Last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines).strip()
            if chunk_content:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    chunk_type="text",
                    name=None,
                    start_line=current_start_line,
                    end_line=len(lines),
                    language=language,
                    docstring=None,
                    metadata={"type": "text"}
                ))
        
        return chunks


# Singleton instance
_chunker = None

def get_chunker() -> MultiLanguageChunker:
    """Get the singleton MultiLanguageChunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = MultiLanguageChunker()
    return _chunker


def detect_repo_language(repo_path: Path) -> str:
    """Convenience function to detect repository language."""
    return LanguageDetector.detect_repo_language(repo_path)
