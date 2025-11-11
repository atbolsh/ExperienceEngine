"""
FAISS backbone for memory tools.
Provides common functionality for querying vector stores.
"""

from typing import List, Optional
from langchain.schema import Document

from vector_store import VectorStoreManager


# Global vector store manager instance
vector_store_manager: Optional[VectorStoreManager] = None


def initialize_vector_store_manager(base_path: str = "."):
    """Initialize the global vector store manager."""
    global vector_store_manager
    vector_store_manager = VectorStoreManager(base_path)
    vector_store_manager.initialize_all_stores()


def query_memory_store(doc_type: str, query: str, k: int = 4) -> str:
    """
    Query a specific memory store and return formatted results.
    
    Args:
        doc_type: Type of memory store ('semantic', 'procedural', or 'episodic')
        query: The search query
        k: Number of results to return
        
    Returns:
        Formatted string with search results
    """
    if vector_store_manager is None:
        return "Error: Vector store manager not initialized."
    
    results = vector_store_manager.query_store(doc_type, query, k)
    
    if not results:
        return f"No relevant information found in {doc_type} memory."
    
    output = f"Found {len(results)} relevant documents in {doc_type} memory:\n\n"
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        output += f"--- Result {i} (Source: {source}) ---\n{doc.page_content}\n\n"
    
    return output


def refresh_all_vector_stores() -> str:
    """
    Refresh all vector stores to pick up new or modified documents.
    Use this after adding or modifying documents in the semantic, procedural, or episodic directories.
    """
    if vector_store_manager is None:
        return "Error: Vector store manager not initialized."
    
    try:
        vector_store_manager.refresh_store("semantic")
        vector_store_manager.refresh_store("procedural")
        vector_store_manager.refresh_store("episodic")
        return "Successfully refreshed all vector stores."
    except Exception as e:
        return f"Error refreshing vector stores: {str(e)}"

