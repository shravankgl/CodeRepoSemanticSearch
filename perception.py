"""
Perception Layer - Extract intent, entities, and tool hints from user input

Uses Claude Haiku to analyze user queries and extract structured information.
"""

import os
import sys
import re
from typing import Optional
from dotenv import load_dotenv
from anthropic import Anthropic

from models import PerceptionResult
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)


def log(stage: str, msg: str):
    """Log to stderr to avoid interfering with JSON communication."""
    sys.stderr.write(f"[{stage}] {msg}\n")
    sys.stderr.flush()


def extract_perception(user_input: str) -> PerceptionResult:
    """
    Extract intent, entities, and tool hints from user input using Claude Haiku.
    
    Args:
        user_input: Raw user input text
        
    Returns:
        PerceptionResult with extracted information
    """
    
    prompt = f"""You are an AI that extracts structured information from user input for a code repository search agent.

The agent can:
1. Clone GitHub repositories
2. Index code for semantic search
3. Search code with natural language
4. Get file contents
5. List files in the repository
6. Get repository status

Analyze this user input: "{user_input}"

Return a JSON object with these fields:
- "intent": One of: "clone_repo", "index_repo", "search_code", "get_file", "list_files", "get_status", "explain", "general_question"
- "entities": List of important entities (repo URL, file paths, search terms, etc.)
- "tool_hint": The most likely MCP tool to use: "clone_repository", "index_repository", "search_code", "get_file_content", "list_files", "get_repo_status", or null
- "repo_url": If a GitHub URL is present, extract it here, otherwise null

Examples:
1. Input: "https://github.com/pallets/flask"
   Output: {{"intent": "clone_repo", "entities": ["github.com/pallets/flask"], "tool_hint": "clone_repository", "repo_url": "https://github.com/pallets/flask"}}

2. Input: "How does authentication work?"
   Output: {{"intent": "search_code", "entities": ["authentication"], "tool_hint": "search_code", "repo_url": null}}

3. Input: "Index the repository"
   Output: {{"intent": "index_repo", "entities": [], "tool_hint": "index_repository", "repo_url": null}}

4. Input: "Show me the main.py file"
   Output: {{"intent": "get_file", "entities": ["main.py"], "tool_hint": "get_file_content", "repo_url": null}}

Return ONLY the JSON object, no additional text or markdown."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.content[0].text.strip()
        log("perception", f"LLM output: {raw}")
        
        # Clean up response (remove markdown if present)
        clean = re.sub(r"^```json\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
        
        try:
            import json
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Try eval as fallback (less safe but handles Python dict format)
            try:
                parsed = eval(clean)
            except Exception as e:
                log("perception", f"Failed to parse: {e}")
                # Return default perception
                return extract_perception_simple(user_input)
        
        # Ensure entities is a list
        if isinstance(parsed.get("entities"), dict):
            parsed["entities"] = list(parsed["entities"].values())
        elif not isinstance(parsed.get("entities"), list):
            parsed["entities"] = []
        
        return PerceptionResult(
            user_input=user_input,
            intent=parsed.get("intent"),
            entities=parsed.get("entities", []),
            tool_hint=parsed.get("tool_hint"),
            repo_url=parsed.get("repo_url")
        )
        
    except Exception as e:
        log("perception", f"Extraction failed: {e}")
        return extract_perception_simple(user_input)


def extract_perception_simple(user_input: str) -> PerceptionResult:
    """
    Simple rule-based perception extraction as fallback.
    
    Args:
        user_input: Raw user input text
        
    Returns:
        PerceptionResult with extracted information
    """
    user_lower = user_input.lower()
    
    # Check for GitHub URL
    github_pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
    github_match = re.search(github_pattern, user_input)
    repo_url = github_match.group(0) if github_match else None
    
    # Determine intent
    intent = "general_question"
    tool_hint = None
    entities = []
    
    if repo_url:
        intent = "clone_repo"
        tool_hint = "clone_repository"
        entities = [repo_url]
    elif any(word in user_lower for word in ["index", "indexing", "build index"]):
        intent = "index_repo"
        tool_hint = "index_repository"
    elif any(word in user_lower for word in ["status", "info", "information"]):
        intent = "get_status"
        tool_hint = "get_repo_status"
    elif any(word in user_lower for word in ["list", "show files", "files in"]):
        intent = "list_files"
        tool_hint = "list_files"
    elif any(word in user_lower for word in ["show", "content", "read", "open"]) and any(ext in user_lower for ext in [".py", ".js", ".ts", ".md"]):
        intent = "get_file"
        tool_hint = "get_file_content"
        # Try to extract file path
        file_pattern = r'[\w\/\-\.]+\.(py|js|ts|md|txt)'
        file_match = re.search(file_pattern, user_input)
        if file_match:
            entities = [file_match.group(0)]
    else:
        # Default to search
        intent = "search_code"
        tool_hint = "search_code"
        # Use the whole input as search query
        entities = [user_input]
    
    return PerceptionResult(
        user_input=user_input,
        intent=intent,
        entities=entities,
        tool_hint=tool_hint,
        repo_url=repo_url
    )
