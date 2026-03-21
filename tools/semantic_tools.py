"""
Semantic memory tools.
Provides tools for querying descriptive information and concepts.
"""

from typing import List

from langchain_compat import BaseModel, Field, StructuredTool, Tool

from tools.memory_tools import query_memory_store


class RAGQueryInput(BaseModel):
    """Input for RAG query tools."""
    query: str = Field(description="The query to search for in the documents")
    k: int = Field(default=4, description="Number of results to return (default: 4)")


def query_semantic_memory(query: str, k: int = 4) -> str:
    """
    Query the semantic memory for descriptive information.
    Use this when you need to understand concepts, definitions, or general knowledge about the environment.
    
    Note: Memories are stored as folders containing text and optionally images (0-10).
    This tool searches only the text content. If a memory mentions images and you need
    to see them, use the image tools (list_memory_images, analyze_memory_image).
    """
    return query_memory_store("semantic", query, k)


def create_semantic_tools() -> List[Tool]:
    """Create and return semantic memory tools."""
    return [
        StructuredTool(
            name="query_semantic_memory",
            func=query_semantic_memory,
            description="Query semantic memory for descriptive information, concepts, and definitions. Use when you need to understand WHAT something is. Memories may contain images - check metadata. Inputs: query (required), k (optional, default 4)",
            args_schema=RAGQueryInput,
        ),
    ]

