"""
Vector store operations for knowledge base
"""
import json
from typing import List, Dict, Any
from pathlib import Path
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self, embeddings, persist_directory: Path):
        """
        Initialize VectorStoreManager
        
        Args:
            embeddings: Embedding function
            persist_directory: Directory to persist vector store
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_vector_store(self, chunks: List[Dict[str, Any]], force_recreate: bool = False):
        """
        Create or load vector store from chunks
        
        Args:
            chunks: List of enriched chunks
            force_recreate: Whether to delete and recreate existing store
        """
        # Delete existing collection if force_recreate
        if force_recreate and self.persist_directory.exists():
            print("Deleting existing vector store...")
            Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            ).delete_collection()
        
        # Convert chunks to LangChain Documents
        print(f"Creating vector store with {len(chunks)} chunks...")
        documents = self._chunks_to_documents(chunks)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        print("✓ Vector store created successfully")
        return self.vectorstore
    
    def load_vector_store(self):
        """Load existing vector store"""
        if not self.persist_directory.exists():
            raise FileNotFoundError(f"Vector store not found at {self.persist_directory}")
        
        print("Loading existing vector store...")
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings
        )
        print("✓ Vector store loaded")
        return self.vectorstore
    
    def get_stats(self) -> Dict[str, int]:
        """Get vector store statistics"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)
        
        return {
            'count': count,
            'dimensions': dimensions
        }
    
    @staticmethod
    def _chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert custom chunks to LangChain Documents"""
        documents = []
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'source': chunk['metadata']['source'],
                    'chunk_id': chunk['metadata']['chunk_id'],
                    'has_images': chunk['metadata']['has_images'],
                    'has_links': chunk['metadata']['has_links'],
                    'num_images': len(chunk['metadata']['images']),
                    'num_links': len(chunk['metadata']['links']),
                    'links_json': json.dumps(chunk['metadata']['links']),
                    'images_json': json.dumps(chunk['metadata']['images']),
                    'raw_text': chunk['raw_text']
                }
            )
            documents.append(doc)
        
        return documents


class ConversationManager:
    """Manages conversational RAG chain"""
    
    def __init__(self, llm, vectorstore, k: int = 40):
        """
        Initialize ConversationManager
        
        Args:
            llm: Language model
            vectorstore: Vector store instance
            k: Number of documents to retrieve
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.k = k
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        self.chain = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup conversational retrieval chain"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get answer
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        result = self.chain.invoke({"question": question})
        return result["answer"]
    
    def reset_memory(self):
        """Clear conversation history"""
        self.memory.clear()
        print("✓ Conversation memory cleared")

