"""
CrewAI Tool Wrapper
===================
Converts LangChain StructuredTool to CrewAI-compatible tools.
"""

from crewai.tools import BaseTool
from typing import Any, Dict, Type
from langchain_core.tools import StructuredTool as LangChainTool
from pydantic import BaseModel, Field


class CrewAIToolWrapper(BaseTool):
    """
    Wraps a LangChain StructuredTool to make it compatible with CrewAI.
    """
    
    # Store the wrapped tool as a class variable pattern
    _wrapped_tool: LangChainTool = None
    
    def __init__(self, langchain_tool: LangChainTool):
        # Get args schema
        args_schema = langchain_tool.args_schema if hasattr(langchain_tool, 'args_schema') else None
        
        # Store tool in a way that Pydantic accepts
        super().__init__(
            name=langchain_tool.name,
            description=langchain_tool.description,
            args_schema=args_schema
        )
        # Store tool after initialization
        object.__setattr__(self, '_wrapped_tool', langchain_tool)
    
    def _run(self, *args, **kwargs) -> Any:
        """Execute the wrapped LangChain tool."""
        tool = getattr(self, '_wrapped_tool', None)
        if not tool:
            raise ValueError("Wrapped tool not found")
        
        # Handle both positional and keyword arguments
        if args and not kwargs:
            # If only positional args, convert first arg to dict if it's a dict
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs = args[0]
            else:
                # Try to use the tool's function signature
                return tool.run(*args)
        return tool.run(kwargs if kwargs else args[0] if args else {})


def wrap_tool(langchain_tool: LangChainTool) -> CrewAIToolWrapper:
    """Convert a LangChain StructuredTool to a CrewAI-compatible tool."""
    return CrewAIToolWrapper(langchain_tool)

