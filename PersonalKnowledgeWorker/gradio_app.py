"""
Gradio web interface for Personal Knowledge Worker
"""
import gradio as gr
from pathlib import Path
from config import get_llm, get_embeddings, VECTOR_DB_PATH
from vector_store import VectorStoreManager, ConversationManager


def setup_app():
    """Setup and return conversation manager"""
    print("Initializing Personal Knowledge Worker...")
    
    # Initialize components
    llm = get_llm()
    embeddings = get_embeddings()
    
    # Load existing vector store
    vector_manager = VectorStoreManager(embeddings, VECTOR_DB_PATH)
    vectorstore = vector_manager.load_vector_store()
    
    # Get stats
    stats = vector_manager.get_stats()
    print(f"✓ Loaded vector store: {stats['count']} vectors")
    
    # Setup conversation manager
    conversation_manager = ConversationManager(llm, vectorstore)
    print("✓ Ready to chat!")
    
    return conversation_manager


# Initialize conversation manager
conversation_manager = setup_app()


def chat(message: str, history):
    """
    Chat function for Gradio interface
    
    Args:
        message: User message
        history: Chat history
        
    Returns:
        Response string
    """
    try:
        response = conversation_manager.ask(message)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def reset_conversation():
    """Reset conversation memory"""
    conversation_manager.reset_memory()
    return "Conversation history cleared!"


# Create Gradio interface
with gr.Blocks(title="Personal Knowledge Worker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Personal Knowledge Worker
    
    Ask questions about your company knowledge base. The system uses RAG to retrieve 
    relevant information from your notes, including image descriptions and links.
    """)
    
    chatbot = gr.ChatInterface(
        fn=chat,
        type="messages",
        examples=[
            "What are the steps to setup AWS 100K env?",
            "How do I configure port forwarding?",
            "What is the RDS configuration for the environment?"
        ],
        title="",
        description="Ask any question about your knowledge base"
    )
    
    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Conversation", variant="secondary")
        reset_output = gr.Textbox(label="Status", visible=False)
    
    reset_btn.click(fn=reset_conversation, outputs=reset_output)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )

