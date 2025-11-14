"""
Episodic memory tools.
Provides tools for querying detailed records of past interactions.
"""

from typing import List

from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

from tools.memory_tools import query_memory_store


class RAGQueryInput(BaseModel):
    """Input for RAG query tools."""
    query: str = Field(description="The query to search for in the documents")
    k: int = Field(default=4, description="Number of results to return (default: 4)")


def query_episodic_memory(query: str, k: int = 4) -> str:
    """
    Query the episodic memory for detailed records of past interactions.
    Use this when you need to recall specific experiences or past events.
    
    Note: Memories are stored as folders containing text and optionally images (0-10).
    This tool searches only the text content. If a memory mentions images and you need
    to see them, use the image tools (list_memory_images, analyze_memory_image).
    """
    return query_memory_store("episodic", query, k)


def create_episodic_tools() -> List[Tool]:
    """Create and return episodic memory tools."""
    return [
        StructuredTool(
            name="query_episodic_memory",
            func=query_episodic_memory,
            description="Query episodic memory for detailed records of past interactions and experiences. Use when you need to recall specific past events. Memories may contain images - check metadata. Inputs: query (required), k (optional, default 4)",
            args_schema=RAGQueryInput,
        ),
    ]

