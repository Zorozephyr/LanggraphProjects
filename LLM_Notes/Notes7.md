# RAG (Retrieval Augmented Generation) - Study Notes

## 📚 Table of Contents
1. [LLM Types](#llm-types)
2. [Vector Embeddings](#vector-embeddings)
3. [RAG Architecture](#rag-architecture)
4. [Document Processing Pipeline](#document-processing-pipeline)
5. [LangChain Components](#langchain-components)
6. [Optimization Strategies](#optimization-strategies)

---

## 🤖 LLM Types

### Auto-Regressive LLMs
- **How it works**: Predicts the **next token** based on **previous tokens**
- **Examples**: GPT-3, GPT-4, Claude
- **Use cases**: Text generation, chat, code completion

### Auto-Encoding LLMs
- **How it works**: Produces output based on the **full input** (bidirectional understanding)
- **Examples**: BERT, OpenAI Embeddings
- **Use cases**: 
  - Sentiment analysis
  - Text classification
  - **Vector embeddings generation** ⭐

---

## 🧮 Vector Embeddings

### What are Vectors?
> Vectors mathematically represent the **semantic meaning** of text as a list of numbers

**Key Properties:**
- **Dimensions**: Typically 384 to 3072 dimensions
- **Semantic similarity**: Similar inputs → Similar vectors (close in vector space)
- **Mathematical operations**: Supports algebraic operations

### Vector Math Example:
```
Vector["King"] - Vector["Man"] + Vector["Woman"] ≈ Vector["Queen"]
```

### Popular Embedding Models:
| Model | Provider | Dimensions | Use Case |
|-------|----------|------------|----------|
| Word2Vec | Google | 300 | Traditional NLP |
| BERT | Google | 768 | General purpose |
| OpenAI Embeddings | OpenAI | 1536 | High accuracy |
| all-MiniLM-L6-v2 | HuggingFace | 384 | Fast & efficient |

---

## 🏗️ RAG Architecture

### High-Level Flow:
```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

User Question
     │
     ▼
┌─────────────────┐
│  Vectorize      │  Convert question to vector embedding
│  Question       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retrieve       │  Find k most similar document chunks
│  from Vector DB │  (Semantic search)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Prompt   │  Question + Retrieved Context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generate   │  Generate answer based on context
│  Answer         │
└────────┬────────┘
         │
         ▼
     Response
```

---

## 📄 Document Processing Pipeline

### Step 1: Load Documents
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all documents from folder
loader = DirectoryLoader('./docs', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
```

### Step 2: Add Metadata
```python
# Add source, date, or other metadata
for doc in documents:
    doc.metadata['source'] = doc.metadata['source']
    doc.metadata['processed_date'] = datetime.now()
```

### Step 3: Split into Chunks
```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # 1000 characters per chunk
    chunk_overlap=200     # 200 characters overlap
)
chunks = text_splitter.split_documents(documents)
```

### Why Chunk Overlap?
```
Chunk 1: [===================>          ]
                           ↓ Overlap
Chunk 2:                [===================>    ]
```
**Benefit**: Ensures context continuity across chunk boundaries

### Visual Example:
```
Original Document (2500 chars):
┌────────────────────────────────────────────────────────┐
│ The quick brown fox jumps over the lazy dog...        │
│ [continues for 2500 characters]                       │
└────────────────────────────────────────────────────────┘

After Chunking (size=1000, overlap=200):
┌──────────────────┐
│ Chunk 1          │ (chars 0-1000)
│ [0──────────1000]│
└──────────────────┘
            └──────┬──────────────┐
                   │ Chunk 2      │ (chars 800-1800)
                   │ [800────1800]│
                   └──────────────┘
                          └──────┬──────────────┐
                                 │ Chunk 3      │ (chars 1600-2500)
                                 │ [1600───2500]│
                                 └──────────────┘
```

---

## 🔧 LangChain Components

### Key Abstractions:

```
┌──────────────────────────────────────────────────────────┐
│                    LangChain RAG Stack                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    │
│  │    LLM     │    │ Retriever  │    │  Memory    │    │
│  ├────────────┤    ├────────────┤    ├────────────┤    │
│  │ GPT-4      │    │ Vector DB  │    │ Chat       │    │
│  │ Claude     │◄───┤ Search     │◄───┤ History    │    │
│  │ Llama      │    │ Top-k      │    │ Context    │    │
│  └────────────┘    └────────────┘    └────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1. LLM (Language Model)
- The brain that generates answers
- Examples: GPT-4, Claude, Llama

### 2. Retriever
- **Purpose**: Fetches relevant document chunks from vector store
- **How**: Converts query to vector → Finds k nearest neighbors
- **Output**: List of relevant text chunks

### 3. Memory
- **Purpose**: Maintains conversation context
- **Types**: 
  - `ConversationBufferMemory` - Stores all messages
  - `ConversationSummaryMemory` - Summarizes old messages

### Putting It Together:
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler

# Setup memory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create conversation chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    callbacks=[StdOutCallbackHandler()]  # Debug: shows all prompts
)

# Use it
result = conversation_chain.invoke({"question": "What is RAG?"})
print(result["answer"])
```

---

## 🎯 Vector Datastores

### Popular Options:

| Database | Type | Best For | Local/Cloud |
|----------|------|----------|-------------|
| **Chroma** | Open source | Development, small scale | Local |
| **FAISS** | Facebook AI | High performance | Local |
| **Pinecone** | Managed | Production, scale | Cloud |
| **Weaviate** | Open source | Production | Both |
| **Qdrant** | Open source | Production | Both |

### Example with Chroma:
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

---

## 🚀 Optimization Strategies

### Problem: Wrong Chunks Retrieved
```
User asks: "How to deploy to production?"
Retrieved: Chunks about development setup ❌
```

### Solution Approaches:

#### 1. **Adjust Chunk Size**
```python
# Too large → Less precise
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Too small → Loss of context
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)

# Sweet spot (depends on use case)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

#### 2. **Increase Chunk Overlap**
```python
# More overlap = Better context continuity
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300  # 30% overlap
)
```

#### 3. **Retrieve More Chunks**
```python
# Default: k=4 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# More context: k=25 chunks (more tokens, better context)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# Balance: k=10 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

#### 4. **Use Full Documents**
Instead of chunking, store entire related documents:
```python
# For smaller, coherent documents
# Don't chunk - use whole document
vectorstore.add_documents(full_documents)
```

#### 5. **Hybrid Search**
Combine semantic search with keyword search:
```
Vector Search (semantic)  +  BM25 (keyword)  =  Better Results
```

---

## 🔍 Debugging RAG with Callbacks

### Using StdOutCallbackHandler:
```python
from langchain_core.callbacks import StdOutCallbackHandler

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    callbacks=[StdOutCallbackHandler()]  # Shows all internal prompts
)
```

**Output Example:**
```
> Entering new ConversationalRetrievalChain chain...
> Entering new LLMChain chain...
Prompt: Use the following pieces of context to answer...
Context: [Retrieved chunks shown here]
Question: What is RAG?
> Finished chain.
```

### Common Problems & Fixes:

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| Wrong chunks retrieved | Check retrieved chunks in callback | Adjust chunking strategy |
| Too generic answers | Not enough context | Increase k value |
| Hallucinations | No relevant chunks found | Improve document coverage |
| Slow responses | Too many chunks | Reduce k value |

---

## 📊 RAG Performance Tuning

### Key Metrics:

```
┌─────────────────────────────────────────┐
│         RAG Optimization Goals          │
├─────────────────────────────────────────┤
│                                         │
│  Relevance ──────┐                     │
│  (Right chunks)   │                     │
│                   ▼                     │
│  Accuracy ───────► Performance         │
│  (Correct answer) │                     │
│                   ▲                     │
│  Speed ───────────┘                     │
│  (Fast response)                        │
│                                         │
└─────────────────────────────────────────┘
```

### Tuning Parameters:

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `chunk_size` | Context per chunk | 500-1500 chars |
| `chunk_overlap` | Context continuity | 10-20% of chunk_size |
| `k` (retrieval) | Context amount | 5-25 chunks |
| `temperature` | Answer creativity | 0.0-0.3 for factual |

---

## 💡 Key Takeaways

1. **RAG = Retrieval + Generation**: Combines document search with LLM generation
2. **Embeddings are crucial**: Quality embeddings = Better retrieval
3. **Chunking strategy matters**: Balance between context and precision
4. **Iterate and optimize**: Use callbacks to debug and improve
5. **It's not magic**: RAG is just vector search + prompt engineering

---

## 🎓 Next Steps

- Experiment with different chunk sizes
- Try various embedding models
- Implement hybrid search
- Add metadata filtering
- Build custom retrievers
- Monitor and improve retrieval accuracy

