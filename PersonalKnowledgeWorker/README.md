# Personal Knowledge Worker

A RAG-based system that processes your OneNote/MHT knowledge base with AI-powered image descriptions and creates an intelligent chatbot.

## Features

- 🖼️ **Image Analysis**: Automatically describes images in your notes using Azure OpenAI Vision
- 🔗 **Link Extraction**: Captures and indexes hyperlinks from your documents
- 💾 **Smart Caching**: Caches processed images to avoid redundant API calls
- 🧠 **RAG System**: Uses vector embeddings for semantic search
- 💬 **Chat Interface**: Gradio-based web UI for natural conversations

## Project Structure

```
PersonalKnowledgeWorker/
├── config.py              # Configuration and Azure setup
├── image_processor.py     # Image extraction and AI description
├── chunk_creator.py       # Text chunking with enrichment
├── vector_store.py        # Vector store and conversation management
├── main.py               # Setup script
├── gradio_app.py         # Web interface
├── requirements.txt      # Dependencies
└── amdocsKnowledgeBase/  # Your knowledge base folder
    ├── Company.mht
    ├── images_cache.pkl  (generated)
    └── knowledge_base_db/ (generated)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   Create a `.env` file with:
   ```
   AUTOX_API_KEY=your_api_key
   NTNET_USERNAME=your_username
   ```

3. **Place your MHT file:**
   Copy your OneNote export to `amdocsKnowledgeBase/Company.mht`

## Usage

### First Time Setup

Run the main setup script to process images and create the vector store:

```bash
python main.py
```

This will:
- Process all images (takes time on first run)
- Create enriched text chunks
- Build the vector database
- Run a test query

### Launch Web Interface

Once setup is complete, launch the Gradio app:

```bash
python gradio_app.py
```

Then open your browser to `http://127.0.0.1:7860`

### Force Reprocessing

To reprocess images or rebuild the database:

```python
# In main.py, modify:
conversation_manager = setup_knowledge_base(
    force_reprocess_images=True,   # Ignore image cache
    force_recreate_vectorstore=True  # Rebuild vector DB
)
```

## How It Works

1. **Image Processing**: Extracts images from MHT, sends to Azure OpenAI Vision API, caches descriptions
2. **Chunk Creation**: Parses HTML sections, matches images, extracts links, creates enriched text
3. **Vector Store**: Embeds chunks using HuggingFace model, stores in ChromaDB
4. **RAG Chain**: Retrieves relevant chunks, feeds to LLM with conversation memory

## Configuration

Edit `config.py` to modify:
- Paths to knowledge base and cache files
- Azure endpoint and model settings
- Embedding model configuration
- Retrieval parameters (k value)

## Notes

- First run takes ~15-30 minutes to process 150 images
- Subsequent runs use cached image descriptions (instant)
- Vector store persists to disk (no need to rebuild)
- Supports conversation history across multiple queries

## Troubleshooting

**Images not being processed:**
- Check Azure API key and endpoint
- Verify proxy settings in config.py
- Check certificate path

**Chat not working:**
- Ensure vector store was created successfully
- Check if `knowledge_base_db/` folder exists
- Verify LLM connectivity

**Out of memory:**
- Reduce retrieval k value in vector_store.py
- Use smaller embedding model
- Process fewer images at once

