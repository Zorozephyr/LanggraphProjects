"""
Main script to setup Personal Knowledge Worker
"""
from pathlib import Path
from config import (
    get_llm, get_embeddings,
    COMPANY_MHT_PATH, IMAGES_CACHE_PATH, VECTOR_DB_PATH
)
from image_processor import ImageProcessor
from chunk_creator import ChunkCreator
from vector_store import VectorStoreManager, ConversationManager


def setup_knowledge_base(force_reprocess_images: bool = False, 
                         force_recreate_vectorstore: bool = False):
    """
    Setup the complete knowledge base pipeline
    
    Args:
        force_reprocess_images: Ignore image cache and reprocess
        force_recreate_vectorstore: Delete and recreate vector store
    """
    print("=" * 70)
    print("Personal Knowledge Worker - Setup")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/5] Initializing LLM and embeddings...")
    llm = get_llm()
    embeddings = get_embeddings()
    print("✓ Initialization complete")
    
    # Process images
    print("\n[2/5] Processing images from MHT file...")
    image_processor = ImageProcessor(llm, IMAGES_CACHE_PATH)
    images = image_processor.extract_and_describe_images(
        COMPANY_MHT_PATH,
        use_cache=not force_reprocess_images
    )
    print(f"✓ Processed {len(images)} images")
    
    # Create chunks
    print("\n[3/5] Creating enriched text chunks...")
    chunks = ChunkCreator.create_enriched_chunks(COMPANY_MHT_PATH, images)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Create vector store
    print("\n[4/5] Building vector store...")
    vector_manager = VectorStoreManager(embeddings, VECTOR_DB_PATH)
    vectorstore = vector_manager.create_vector_store(
        chunks,
        force_recreate=force_recreate_vectorstore
    )
    
    # Show stats
    stats = vector_manager.get_stats()
    print(f"✓ Vector store ready: {stats['count']} vectors, {stats['dimensions']} dimensions")
    
    # Setup conversation manager
    print("\n[5/5] Setting up conversation chain...")
    conversation_manager = ConversationManager(llm, vectorstore)
    print("✓ Conversation chain ready")
    
    print("\n" + "=" * 70)
    print("Setup Complete! Ready to answer questions.")
    print("=" * 70)
    
    return conversation_manager


def main():
    """Main entry point"""
    # Setup knowledge base
    conversation_manager = setup_knowledge_base(
        force_reprocess_images=False,  # Set to True to ignore cache
        force_recreate_vectorstore=False  # Set to True to rebuild DB
    )
    
    # Example query
    print("\n" + "=" * 70)
    print("Testing with example query...")
    print("=" * 70)
    
    query = "What are the steps to setup AWS 100K env?"
    print(f"\nQuestion: {query}")
    print("\nAnswer:")
    print("-" * 70)
    answer = conversation_manager.ask(query)
    print(answer)
    print("-" * 70)


if __name__ == "__main__":
    main()

