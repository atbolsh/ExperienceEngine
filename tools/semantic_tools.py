"""
Semantic memory tools.
Provides tools for querying descriptive information and concepts.
"""

from typing import List

from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field

from tools.memory_tools import query_memory_store


class RAGQueryInput(BaseModel):
    """Input for RAG query tools."""
    query: str = Field(description="The query to search for in the documents")
    k: int = Field(default=4, description="Number of results to return (default: 4)")


def query_semantic_memory(query: str, k: int = 4) -> str:
    """
    Query the semantic memory for descriptive information.
    Use this when you need to understand concepts, definitions, or general knowledge about the environment.
    """
    return query_memory_store("semantic", query, k)


def create_semantic_tools() -> List[Tool]:
    """Create and return semantic memory tools."""
    return [
        Tool(
            name="query_semantic_memory",
            func=query_semantic_memory,
            description="Query semantic memory for descriptive information, concepts, and definitions. Use when you need to understand WHAT something is. Inputs: query (required), k (optional, default 4)",
            args_schema=RAGQueryInput,
        ),
    ]

