"""
Decision Layer - Generate plans using Claude Haiku

Takes perception result and memory context to decide:
- Which tool to call (FUNCTION_CALL format)
- Or final answer (FINAL_ANSWER format)
"""

import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from anthropic import Anthropic

from models import PerceptionResult, MemoryItem
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)


def log(stage: str, msg: str):
    """Log to stderr to avoid interfering with JSON communication."""
    sys.stderr.write(f"[{stage}] {msg}\n")
    sys.stderr.flush()


def generate_plan(
    perception: PerceptionResult,
    memory_context: str = "",
    tool_descriptions: Optional[str] = None,
    search_results: Optional[str] = None
) -> str:
    """
    Generate a plan (tool call or final answer) using Claude Haiku.
    
    Args:
        perception: Analyzed user input with intent and entities
        memory_context: Context from memory (conversation history, repo state)
        tool_descriptions: Available MCP tools
        search_results: Results from previous code searches
        
    Returns:
        Either "FUNCTION_CALL: tool_name|param=value" or "FINAL_ANSWER: [answer]"
    """
    
    tool_context = f"\nAvailable tools:\n{tool_descriptions}" if tool_descriptions else ""
    search_context = f"\nPrevious search results:\n{search_results}" if search_results else ""
    
    prompt = f"""You are a code repository search agent. Your job is to help users understand codebases by:
1. Cloning GitHub repositories
2. Indexing code for semantic search
3. Answering questions about the code

{tool_context}

Current Context:
{memory_context}
{search_context}

User Input Analysis:
- Raw Input: "{perception.user_input}"
- Detected Intent: {perception.intent}
- Entities: {', '.join(perception.entities) if perception.entities else 'None'}
- Tool Hint: {perception.tool_hint or 'None'}
- Repository URL: {perception.repo_url or 'None'}

Instructions:
1. If a tool needs to be called, respond with EXACTLY:
   FUNCTION_CALL: tool_name|param1=value1|param2=value2

2. If you can provide a final answer (e.g., after receiving search results), respond with:
   FINAL_ANSWER: [Your detailed answer based on the code context]

3. Tool parameter formats:
   - clone_repository|repo_url=https://github.com/user/repo
   - index_repository
   - search_code|query=search terms|top_k=5
   - get_file_content|file_path=path/to/file.py
   - list_files|pattern=*.py
   - get_repo_status

4. Decision logic:
   - If intent is "clone_repo" and repo_url is present → call clone_repository
   - If intent is "index_repo" → call index_repository
   - If intent is "search_code" → call search_code with the query
   - If intent is "get_file" → call get_file_content
   - If intent is "list_files" → call list_files
   - If intent is "get_status" → call get_repo_status
   - If search results are available and sufficient → provide FINAL_ANSWER

5. For FINAL_ANSWER:
   - Synthesize information from search results
   - Cite specific files and line numbers when available
   - Explain code concepts clearly
   - If information is insufficient, suggest searching for more specific terms

Respond with ONLY one line: FUNCTION_CALL: ... OR FINAL_ANSWER: [...]
Do NOT include any other text or explanation."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.content[0].text.strip()
        log("decision", f"LLM output: {raw}")
        
        # Extract the function call or final answer line
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("FUNCTION_CALL:") or line.startswith("FINAL_ANSWER:"):
                return line
        
        # If no valid format found, return the raw response
        return raw
        
    except Exception as e:
        log("decision", f"Plan generation failed: {e}")
        return "FINAL_ANSWER: [I encountered an error while processing your request. Please try again.]"


def generate_explanation(code_snippet: str, context: str = "") -> str:
    """
    Generate a natural language explanation of code.
    
    Args:
        code_snippet: The code to explain
        context: Additional context about the code
        
    Returns:
        Human-readable explanation
    """
    
    prompt = f"""You are a code explanation assistant. Explain the following code in clear, concise terms.

{f"Context: {context}" if context else ""}

Code:
```
{code_snippet[:3000]}
```

Provide a clear explanation that covers:
1. What the code does (high-level purpose)
2. Key components and their roles
3. Important patterns or techniques used
4. Any notable design decisions

Keep the explanation concise but informative."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
        
    except Exception as e:
        log("decision", f"Explanation generation failed: {e}")
        return f"Failed to generate explanation: {e}"


def synthesize_answer(query: str, search_results: List[dict], memory_context: str = "") -> str:
    """
    Synthesize a comprehensive answer from search results.
    
    Args:
        query: Original user question
        search_results: Code search results
        memory_context: Additional context from memory
        
    Returns:
        Synthesized answer with citations
    """
    
    # Format search results for the prompt
    results_text = ""
    for i, result in enumerate(search_results, 1):
        results_text += f"""
[{i}] File: {result.get('file_path', 'unknown')}
    Type: {result.get('type', 'unknown')} | Name: {result.get('name', 'N/A')}
    Lines: {result.get('lines', 'N/A')}
    Content:
    ```
    {result.get('content', '')[:800]}
    ```
"""
    
    prompt = f"""You are a code analysis assistant. Answer the user's question based on the code search results.

Question: {query}

{f"Context: {memory_context}" if memory_context else ""}

Search Results:
{results_text}

Instructions:
1. Synthesize a clear, comprehensive answer
2. Reference specific files and functions using [1], [2], etc.
3. Explain code concepts in plain English
4. If the results don't fully answer the question, mention what additional information might help
5. Keep the answer focused and well-structured

Provide your answer:"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
        
    except Exception as e:
        log("decision", f"Answer synthesis failed: {e}")
        return f"Failed to synthesize answer: {e}"
