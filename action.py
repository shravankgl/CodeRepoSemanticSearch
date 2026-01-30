"""
Action Layer - Execute MCP tools based on decision layer plans

Parses FUNCTION_CALL format and executes tools via MCP session.
"""

import sys
import ast
import json
from typing import Dict, Any, List, Optional, Tuple
from mcp import ClientSession

from models import ToolCallResult


def log(stage: str, msg: str):
    """Log to stderr to avoid interfering with JSON communication."""
    sys.stderr.write(f"[{stage}] {msg}\n")
    sys.stderr.flush()


def parse_function_call(response: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse FUNCTION_CALL string into tool name and arguments.
    
    Format: FUNCTION_CALL: tool_name|param1=value1|param2=value2
    
    Args:
        response: The FUNCTION_CALL string from decision layer
        
    Returns:
        Tuple of (tool_name, arguments_dict)
    """
    try:
        if not response.startswith("FUNCTION_CALL:"):
            raise ValueError("Not a valid FUNCTION_CALL")
        
        # Extract everything after "FUNCTION_CALL:"
        _, function_info = response.split(":", 1)
        function_info = function_info.strip()
        
        # Split by pipe character
        parts = [p.strip() for p in function_info.split("|")]
        
        # First part is function name
        func_name = parts[0]
        param_parts = parts[1:]
        
        # Parse parameters
        result = {}
        for part in param_parts:
            if not part:
                continue
            if "=" not in part:
                log("parser", f"Skipping invalid param: {part}")
                continue
                
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            # Try to parse the value
            try:
                # Try JSON first
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    # Try Python literal
                    parsed_value = ast.literal_eval(value)
                except Exception:
                    # Keep as string
                    parsed_value = value
            
            # Handle nested keys (e.g., input.string)
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = parsed_value
        
        log("parser", f"Parsed: {func_name} → {result}")
        return func_name, result
        
    except Exception as e:
        log("parser", f"Failed to parse FUNCTION_CALL: {e}")
        raise


async def execute_tool(
    session: ClientSession, 
    tools: List[Any], 
    response: str
) -> ToolCallResult:
    """
    Execute a tool via MCP session.
    
    Args:
        session: MCP ClientSession
        tools: List of available tools
        response: FUNCTION_CALL string from decision layer
        
    Returns:
        ToolCallResult with execution results
    """
    try:
        tool_name, arguments = parse_function_call(response)
        
        # Verify tool exists
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registered tools")
        
        log("action", f"Calling '{tool_name}' with: {arguments}")
        
        # Execute the tool
        result = await session.call_tool(tool_name, arguments=arguments)
        
        # Extract result content
        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                out = [getattr(item, 'text', str(item)) for item in result.content]
                # If single item, unwrap
                if len(out) == 1:
                    out = out[0]
            else:
                out = getattr(result.content, 'text', str(result.content))
        else:
            out = str(result)
        
        log("action", f"'{tool_name}' returned successfully")
        
        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            result=out,
            success=True,
            error=None
        )
        
    except Exception as e:
        log("action", f"Execution failed: {e}")
        
        # Try to extract tool name for the error result
        try:
            tool_name, arguments = parse_function_call(response)
        except Exception:
            tool_name = "unknown"
            arguments = {}
        
        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            result=None,
            success=False,
            error=str(e)
        )


def format_tool_result(result: ToolCallResult) -> str:
    """
    Format a tool result for display or LLM context.
    
    Args:
        result: ToolCallResult from execute_tool
        
    Returns:
        Formatted string representation
    """
    if result.success:
        if isinstance(result.result, str):
            try:
                # Try to parse as JSON for pretty printing
                parsed = json.loads(result.result)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                return result.result
        else:
            return json.dumps(result.result, indent=2, default=str)
    else:
        return f"Error: {result.error}"


def extract_search_results(tool_result: ToolCallResult) -> Optional[List[dict]]:
    """
    Extract search results from a search_code tool result.
    
    Args:
        tool_result: Result from search_code tool
        
    Returns:
        List of result dictionaries or None
    """
    if tool_result.tool_name != "search_code" or not tool_result.success:
        return None
    
    try:
        if isinstance(tool_result.result, str):
            data = json.loads(tool_result.result)
        else:
            data = tool_result.result
        
        return data.get("results", [])
    except Exception:
        return None
