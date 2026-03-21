"""
Stable import paths for LangChain 0.2.x and 1.x.

In 1.x, Tool / StructuredTool / Document / HumanMessage / etc. live under langchain-core
(and RecursiveCharacterTextSplitter under langchain-text-splitters), not langchain.*.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool, StructuredTool, tool

try:
    from langchain_core.pydantic_v1 import BaseModel, Field, validator
except ImportError:
    # Newer langchain-core drops langchain_core.pydantic_v1; Pydantic v2 ships a v1-compat namespace.
    from pydantic.v1 import BaseModel, Field, validator

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # LangChain >=1.x
    from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__ = [
    "BaseModel",
    "Document",
    "Field",
    "HumanMessage",
    "RecursiveCharacterTextSplitter",
    "StructuredTool",
    "Tool",
    "tool",
    "validator",
]
