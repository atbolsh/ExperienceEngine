"""
Working memory query tools.
Provides access to temporary working memory that is cleared at session end.
"""

from typing import List
from langchain.tools import Tool

from tools.memory_tools import get_vector_store_manager


def query_working_memory(query: str) -> str:
    """
    Query the working memory (temporary session memory).
    
    Working memory is for temporary information during the current session,
    such as recent observations, intermediate calculations, or temporary images.
    This memory is automatically cleared at the end of each session.
    
    Args:
        query: The search query
        
    Returns:
        Formatted results from working memory
    """
    manager = get_vector_store_manager()
    
    if manager is None:
        return "Error: Vector store manager not initialized."
    
    if manager.working_store is None:
        return "Working memory is empty or not yet initialized."
    
    results = manager.query_store("working", query, k=4)
    
    if not results:
        return "No relevant information found in working memory."
    
    formatted_results = []
    for i, doc in enumerate(results, 1):
        result_text = f"Result {i}:\n"
        result_text += f"Content: {doc.page_content}\n"
        
        # Include metadata
        if doc.metadata:
            result_text += "Metadata:\n"
            for key, value in doc.metadata.items():
                if key not in ['has_images', 'image_count']:
                    result_text += f"  {key}: {value}\n"
            
            # Mention images if they exist
            if doc.metadata.get('has_images', False):
                image_names = doc.metadata.get('image_names', [])
                result_text += f"  Images available: {', '.join(image_names)}\n"
        
        formatted_results.append(result_text)
    
    return "\n---\n".join(formatted_results)


def create_working_tools() -> List[Tool]:
    """
    Create working memory query tools.
    
    Returns:
        List of working memory tools
    """
    return [
        Tool(
            name="query_working_memory",
            func=query_working_memory,
            description=(
                "Search through working memory (temporary session memory). "
                "Working memory stores temporary information like recent game states, "
                "observations, and intermediate results. It is automatically cleared "
                "at the end of each session. Use this to recall recent observations or "
                "previously captured images from the current session."
            )
        ),
    ]

