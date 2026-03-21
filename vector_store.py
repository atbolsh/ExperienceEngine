"""
Vector store management for FAISS indices.
Handles creation, loading, and querying of FAISS vector databases for each document type.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_compat import Document, RecursiveCharacterTextSplitter


class VectorStoreManager:
    """Manages FAISS vector stores for semantic, procedural, and episodic documents."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector stores
        self.semantic_store: Optional[FAISS] = None
        self.procedural_store: Optional[FAISS] = None
        self.episodic_store: Optional[FAISS] = None
        self.working_store: Optional[FAISS] = None
        
        # Paths for persisted indices
        self.semantic_path = self.base_path / "semantic"
        self.procedural_path = self.base_path / "procedural"
        self.episodic_path = self.base_path / "episodic"
        self.working_path = self.base_path / "working"
        
        self.semantic_index_path = self.base_path / ".faiss_semantic"
        self.procedural_index_path = self.base_path / ".faiss_procedural"
        self.episodic_index_path = self.base_path / ".faiss_episodic"
        self.working_index_path = self.base_path / ".faiss_working"
    
    def load_documents_from_directory(self, directory: Path) -> List[Document]:
        """
        Load all text documents from memory folders in a directory.
        
        Each memory is a folder containing:
        - Exactly one text file (.txt or .md)
        - 0-10 images (which are ignored for indexing)
        
        Only the text content is loaded and indexed in FAISS.
        """
        if not directory.exists():
            return []
        
        from tools.memory_schema import list_memory_folders
        
        documents = []
        
        # Load memory folders
        memory_folders = list_memory_folders(directory)
        
        for memory in memory_folders:
            try:
                # Create metadata including memory name and available images
                metadata = {
                    "source": str(memory.text_file),
                    "memory_name": memory.memory_name,
                    "memory_folder": str(memory.folder_path),
                    "has_images": len(memory.image_files) > 0,
                    "image_count": len(memory.image_files),
                    "image_names": memory.image_names
                }
                
                documents.append(Document(
                    page_content=memory.text_content,
                    metadata=metadata
                ))
            except Exception as e:
                print(f"Error loading memory {memory.folder_name}: {e}")
        
        return documents
    
    def create_or_load_vector_store(self, doc_type: str) -> Optional[FAISS]:
        """Create or load a FAISS vector store for a specific document type."""
        if doc_type == "semantic":
            directory = self.semantic_path
            index_path = self.semantic_index_path
        elif doc_type == "procedural":
            directory = self.procedural_path
            index_path = self.procedural_index_path
        elif doc_type == "episodic":
            directory = self.episodic_path
            index_path = self.episodic_index_path
        elif doc_type == "working":
            directory = self.working_path
            index_path = self.working_index_path
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        # Try to load existing index
        if index_path.exists():
            try:
                vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing {doc_type} vector store from {index_path}")
                return vector_store
            except Exception as e:
                print(f"Error loading existing index for {doc_type}: {e}")
        
        # Create new index from documents
        documents = self.load_documents_from_directory(directory)
        
        if not documents:
            print(f"No documents found in {directory}. Creating empty vector store.")
            return None
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        if not chunks:
            return None
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the index
        vector_store.save_local(str(index_path))
        print(f"Created and saved {doc_type} vector store with {len(chunks)} chunks")
        
        return vector_store
    
    def initialize_all_stores(self):
        """Initialize all vector stores including working memory."""
        print("Initializing vector stores...")
        self.semantic_store = self.create_or_load_vector_store("semantic")
        self.procedural_store = self.create_or_load_vector_store("procedural")
        self.episodic_store = self.create_or_load_vector_store("episodic")
        self.working_store = self.create_or_load_vector_store("working")
        print("Vector stores initialized.")
    
    def refresh_store(self, doc_type: str):
        """Refresh a specific vector store by rebuilding it from documents."""
        if doc_type == "semantic":
            index_path = self.semantic_index_path
        elif doc_type == "procedural":
            index_path = self.procedural_index_path
        elif doc_type == "episodic":
            index_path = self.episodic_index_path
        elif doc_type == "working":
            index_path = self.working_index_path
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        # Remove old index if it exists
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)
        
        # Create new index
        new_store = self.create_or_load_vector_store(doc_type)
        
        # Update the instance variable
        if doc_type == "semantic":
            self.semantic_store = new_store
        elif doc_type == "procedural":
            self.procedural_store = new_store
        elif doc_type == "episodic":
            self.episodic_store = new_store
        elif doc_type == "working":
            self.working_store = new_store
    
    def query_store(self, doc_type: str, query: str, k: int = 4) -> List[Document]:
        """Query a specific vector store and return relevant documents."""
        if doc_type == "semantic":
            store = self.semantic_store
        elif doc_type == "procedural":
            store = self.procedural_store
        elif doc_type == "episodic":
            store = self.episodic_store
        elif doc_type == "working":
            store = self.working_store
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        if store is None:
            return []
        
        try:
            results = store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error querying {doc_type} store: {e}")
            return []
    
    def clear_working_memory(self):
        """Clear all contents of working memory directory and refresh the store."""
        import shutil
        
        # Clear the working directory
        if self.working_path.exists():
            for item in self.working_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print(f"Cleared working memory directory: {self.working_path}")
        
        # Clear the working index
        if self.working_index_path.exists():
            shutil.rmtree(self.working_index_path)
        
        # Reset the working store
        self.working_store = None
        print("Working memory cleared.")

