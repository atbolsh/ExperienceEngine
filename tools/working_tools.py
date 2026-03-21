"""
Working memory tools.
Provides tools for querying temporary working memory that gets cleared at session end.
"""

from typing import List

from langchain_compat import BaseModel, Field, StructuredTool, Tool

from tools.memory_tools import query_memory_store


class RAGQueryInput(BaseModel):
    """Input for RAG query tools."""
    query: str = Field(description="The query to search for in the documents")
    k: int = Field(default=4, description="Number of results to return (default: 4)")


def query_working_memory(query: str, k: int = 4) -> str:
    """
    Query the working memory for temporary information from the current session.
    Use this to recall recent observations, captured images, and session-specific notes.
    
    NOTE: Working memory is automatically cleared at the end of each session.
    Store important information in semantic, procedural, or episodic memory for long-term retention.
    
    Working memory is ideal for:
    - Recent camera captures and observations
    - Temporary notes during exploration
    - Session-specific data that doesn't need long-term storage
    """
    return query_memory_store("working", query, k)


def create_working_tools() -> List[Tool]:
    """Create and return working memory tools."""
    return [
        StructuredTool(
            name="query_working_memory",
            func=query_working_memory,
            description="Query working memory for temporary session information like recent observations and captures. This memory is cleared at session end. Inputs: query (required), k (optional, default 4)",
            args_schema=RAGQueryInput,
        ),
    ]

