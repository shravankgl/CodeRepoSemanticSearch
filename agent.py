"""
Code Repository Semantic Search - Main Agent Entry Point

Interactive CLI agent that implements:
- Perception → Decision → Action loop
- RAG with FAISS vector memory
- MCP tools for repository operations

Usage:
    python agent.py
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MAX_AGENT_STEPS, BASE_DIR
from perception import extract_perception
from decision import generate_plan, synthesize_answer
from action import execute_tool, format_tool_result, extract_search_results
from memory import MemoryManager

load_dotenv()

console = Console()


class RateLimiter:
    """Simple rate limiter for Claude API."""
    
    def __init__(self, max_requests_per_minute: int = 5):
        self.max_requests = max_requests_per_minute
        self.request_times = []
    
    async def acquire(self):
        current_time = time.time()
        
        # Remove old request times
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we need to wait
        if len(self.request_times) >= self.max_requests:
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request) + 1
            if wait_time > 0:
                console.print(f"[yellow]⏳ Rate limit reached. Waiting {wait_time:.1f}s...[/yellow]")
                await asyncio.sleep(wait_time)
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        self.request_times.append(current_time)


rate_limiter = RateLimiter(max_requests_per_minute=5)


def get_tool_descriptions(tools) -> str:
    """Get formatted tool descriptions for the decision layer."""
    descriptions = []
    for tool in tools:
        desc = getattr(tool, 'description', 'No description')
        descriptions.append(f"- {tool.name}: {desc}")
    return "\n".join(descriptions)


async def agent_loop(session: ClientSession, tools: list, memory: MemoryManager):
    """
    Main agent interaction loop.
    
    Implements: Perception → Decision → Action
    """
    
    tool_descriptions = get_tool_descriptions(tools)
    
    console.print(Panel(
        "[bold cyan]Code Repository Semantic Search Agent[/bold cyan]\n\n"
        "Commands:\n"
        "  • Paste a GitHub URL to clone and index a repository\n"
        "  • Ask questions about the code\n"
        "  • Type 'status' to check repository status\n"
        "  • Type 'exit' or 'quit' to exit\n",
        title="🔍 Welcome",
        border_style="cyan"
    ))
    
    while True:
        try:
            # Get user input
            console.print()
            user_input = console.input("[bold green]You:[/bold green] ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            # Store user message
            memory.add_message("user", user_input)
            
            # ========== PERCEPTION ==========
            console.print("[dim]🧠 Analyzing input...[/dim]")
            await rate_limiter.acquire()
            perception = extract_perception(user_input)
            
            console.print(f"[dim]   Intent: {perception.intent} | Tool hint: {perception.tool_hint}[/dim]")
            
            # ========== AGENT LOOP ==========
            step = 0
            last_search_results = None
            
            while step < MAX_AGENT_STEPS:
                step += 1
                
                # Get memory context
                memory_context = memory.get_context_for_llm()
                
                # Format search results if available
                search_results_str = None
                if last_search_results:
                    search_results_str = json.dumps(last_search_results, indent=2)
                
                # ========== DECISION ==========
                console.print(f"[dim]🤔 Planning (step {step}/{MAX_AGENT_STEPS})...[/dim]")
                await rate_limiter.acquire()
                
                plan = generate_plan(
                    perception=perception,
                    memory_context=memory_context,
                    tool_descriptions=tool_descriptions,
                    search_results=search_results_str
                )
                
                # ========== ACTION ==========
                if plan.startswith("FUNCTION_CALL:"):
                    # Extract function name for display
                    func_info = plan.split(":", 1)[1].strip()
                    func_name = func_info.split("|")[0]
                    
                    console.print(f"[cyan]🔧 Calling tool: {func_name}[/cyan]")
                    
                    result = await execute_tool(session, tools, plan)
                    
                    if result.success:
                        # Parse and display result
                        formatted = format_tool_result(result)
                        
                        # Show abbreviated result
                        try:
                            parsed = json.loads(formatted)
                            if parsed.get("success"):
                                console.print(f"[green]✓ {parsed.get('message', 'Success')}[/green]")
                            else:
                                console.print(f"[red]✗ {parsed.get('message', 'Failed')}[/red]")
                        except:
                            console.print(f"[green]✓ Tool executed[/green]")
                        
                        # Store tool output in memory
                        memory.add_tool_output(result.tool_name, result.arguments, result.result)
                        
                        # Update repo state if needed
                        if result.tool_name == "clone_repository":
                            try:
                                data = json.loads(result.result)
                                if data.get("success"):
                                    memory.set_repo_state(
                                        repo_url=perception.repo_url,
                                        repo_name=data.get("repo_name"),
                                        local_path=data.get("local_path"),
                                        is_cloned=True,
                                        file_count=data.get("file_count", 0)
                                    )
                                    # Initialize vector memory for this repo
                                    memory.init_vector_memory(data.get("repo_name"))
                            except:
                                pass
                        
                        elif result.tool_name == "index_repository":
                            try:
                                data = json.loads(result.result)
                                if data.get("success"):
                                    memory.set_repo_state(
                                        is_indexed=True,
                                        chunk_count=data.get("chunk_count", 0)
                                    )
                            except:
                                pass
                        
                        elif result.tool_name == "search_code":
                            # Extract search results for answer synthesis
                            last_search_results = extract_search_results(result)
                        
                    else:
                        console.print(f"[red]✗ Tool error: {result.error}[/red]")
                        break
                    
                elif plan.startswith("FINAL_ANSWER:"):
                    # Extract and display final answer
                    answer = plan.replace("FINAL_ANSWER:", "").strip()
                    
                    # If we have search results and answer is generic, synthesize better answer
                    if last_search_results and len(answer) < 100:
                        console.print("[dim]📝 Synthesizing answer from search results...[/dim]")
                        await rate_limiter.acquire()
                        answer = synthesize_answer(
                            user_input, 
                            last_search_results,
                            memory_context
                        )
                    
                    # Clean up answer formatting
                    answer = answer.strip("[]")
                    
                    console.print()
                    console.print(Panel(
                        Markdown(answer),
                        title="🤖 Assistant",
                        border_style="blue"
                    ))
                    
                    # Store assistant message
                    memory.add_message("assistant", answer)
                    break
                    
                else:
                    # Invalid format - try to use as final answer
                    console.print()
                    console.print(Panel(
                        plan,
                        title="🤖 Assistant",
                        border_style="blue"
                    ))
                    memory.add_message("assistant", plan)
                    break
            
            if step >= MAX_AGENT_STEPS:
                console.print("[yellow]⚠️ Reached maximum steps. Please try a more specific query.[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


async def main():
    """Main entry point."""
    
    console.print("[dim]Starting Code Repository Semantic Search Agent...[/dim]")
    
    # Initialize memory
    memory = MemoryManager()
    
    # Setup MCP server parameters
    server_script = BASE_DIR / "repo_tools.py"
    
    if not server_script.exists():
        console.print(f"[red]Error: repo_tools.py not found at {server_script}[/red]")
        return
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_script)],
        cwd=str(BASE_DIR)
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            console.print("[dim]Connected to MCP server...[/dim]")
            
            async with ClientSession(read, write) as session:
                await session.initialize()
                console.print("[dim]MCP session initialized...[/dim]")
                
                # Get available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                console.print(f"[dim]Loaded {len(tools)} tools: {', '.join(t.name for t in tools)}[/dim]")
                
                # Run the agent loop
                await agent_loop(session, tools, memory)
                
    except Exception as e:
        console.print(f"[red]Failed to start agent: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
