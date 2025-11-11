"""
Procedural memory tools.
Provides tools for querying step-by-step instructions and how-to guides.
"""

from typing import List

from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field

from tools.memory_tools import query_memory_store


class RAGQueryInput(BaseModel):
    """Input for RAG query tools."""
    query: str = Field(description="The query to search for in the documents")
    k: int = Field(default=4, description="Number of results to return (default: 4)")


def query_procedural_memory(query: str, k: int = 4) -> str:
    """
    Query the procedural memory for step-by-step instructions.
    Use this when you need to know HOW to do something or what steps to follow.
    
    Note: Memories are stored as folders containing text and optionally images (0-10).
    This tool searches only the text content. If a memory mentions images and you need
    to see them, use the image tools (list_memory_images, analyze_memory_image).
    """
    return query_memory_store("procedural", query, k)


def create_procedural_tools() -> List[Tool]:
    """Create and return procedural memory tools."""
    return [
        Tool(
            name="query_procedural_memory",
            func=query_procedural_memory,
            description="Query procedural memory for step-by-step instructions and how-to guides. Use when you need to know HOW to do something. Memories may contain images - check metadata. Inputs: query (required), k (optional, default 4)",
            args_schema=RAGQueryInput,
        ),
    ]

