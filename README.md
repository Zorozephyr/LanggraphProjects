# ChatWithMeAgent: AI Agent Projects Suite

## Overview

This repository contains three advanced AI agent projects demonstrating different approaches to building intelligent conversational systems with tool integration, state management, and multi-agent patterns.

### 🔍 Quick Comparison

| Feature | Career Agent | Sidekick Assistant | Dataset Generator |
|---------|--------------|-------------------|-------------------|
| **Primary Use** | Career chatbot | Research & automation | Data generation |
| **Interface** | Gradio web UI | Gradio web UI | CLI (Command-line) |
| **Architecture** | Simple tool integration | Multi-agent with evaluator | Iterative with validation |
| **Key Capability** | Answer questions + record leads | Browse web, execute code, plan trips | Generate synthetic datasets |
| **User Input** | Natural language chat | Natural language chat | Structured prompts |
| **Output** | Conversational responses | Task completion + artifacts | JSON datasets (50 records) |
| **Feedback Loop** | None | Evaluator retries | Quality-based regeneration |
| **Deployment** | HuggingFace Spaces | Cloud/Docker | Local/Scheduled script |
| **Complexity** | ⭐ Beginner | ⭐⭐⭐ Advanced | ⭐⭐ Intermediate |

---

## 📋 Table of Contents

1. [Project 1: Career Conversation Agent (ChatWithPushNotifications)](#project-1-career-conversation-agent)
2. [Project 2: Sidekick Multi-Agent Assistant](#project-2-sidekick-multi-agent-assistant)
3. [Project 3: Synthetic Dataset Generator](#project-3-synthetic-dataset-generator)
4. [Installation & Setup](#installation--setup)
5. [Architecture Patterns](#architecture-patterns)
6. [Deployment](#deployment)

---

## Project 1: Career Conversation Agent

### What It Does

A specialized Gradio-based chatbot that impersonates a professional (powered by your LinkedIn profile and summary). The agent:

- **Answers questions** about your background, skills, and experience
- **Records user details** when people express interest in connecting
- **Tracks unanswered questions** for follow-up
- **Sends push notifications** for each user interaction
- **Uses OpenAI tools** to intelligently manage conversations

### Key Features

✅ **Azure OpenAI Integration** - Uses GPT-4o with custom authentication  
✅ **Tool Integration** - Two custom tools:
   - `record_user_details`: Captures interested users' email and notes
   - `record_unknown_question`: Records questions you can't answer

✅ **Push Notifications** - Integrates with Pushover API for real-time alerts  
✅ **PDF Resume Parsing** - Reads LinkedIn profile as PDF  
✅ **Simple Web Interface** - Built with Gradio, easy to chat with  
✅ **Deployable** - Ready for HuggingFace Spaces deployment  

### Files

```
📁 1_foundations/
├── ChatWithPushNotifications.ipynb  (Main notebook)
├── app.py                           (Deployment script)
├── me/
│   ├── linkedin.pdf                 (Your LinkedIn profile)
│   └── summary.txt                  (Your professional summary)
└── requirements.txt
```

**📂 [View Project 1 Files →](./1_foundations/)**

### Quick Start

```bash
# Navigate to project
cd 1_foundations

# Install dependencies (from root)
cd ..
uv sync
cd 1_foundations

# Run the notebook
jupyter notebook ChatWithPushNotifications.ipynb

# Or run the app directly
python app.py
```

### Environment Setup

Create a `.env` file with:
```
OPENAI_API_KEY=sk-...
AUTOX_API_KEY=...  (if using Azure)
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
NTNET_USERNAME=...
```

### How It Works

```
User Question
    ↓
LLM with Tools (GPT-4o)
    ↓
    ├─ If has answer → Return response
    ├─ If user interested → Call record_user_details tool
    └─ If question unanswered → Call record_unknown_question tool
    ↓
Push Notification Sent
    ↓
Response Displayed
```

### Deploy to HuggingFace Spaces

```bash
uv tool install 'huggingface_hub[cli]'
hf auth login
cd 1_foundations
uv run gradio deploy
```

Follow the prompts to:
1. Name it "career_conversation"
2. Specify `app.py` as entry point
3. Choose CPU-basic hardware
4. Provide your secrets (API keys, Pushover credentials)

---

## Project 2: Sidekick Multi-Agent Assistant

### What It Does

A sophisticated LangGraph-based AI agent that can:

- **Research information** using web search and Wikipedia
- **Browse websites** using Playwright with color-coded location markers
- **Plan trips** with interactive Google Maps
- **Execute Python code** for data analysis
- **Manage files** in a sandboxed directory
- **Provide feedback loops** - agents improve through evaluator feedback
- **Maintain conversation memory** with persistent state

### Key Features

✅ **Multi-Agent Pattern** - Worker, Tools, and Evaluator nodes  
✅ **Feedback Loops** - Agents learn and improve from evaluator feedback  
✅ **7 Built-in Tools**:
   - Playwright browser automation (navigate, extract text, click, get links)
   - Web search (Google Serper)
   - Wikipedia queries
   - Python code execution (REPL)
   - File management (read/write/create)
   - Push notifications
   - **Trip planning with Google Maps** (NEW!)

✅ **Structured Outputs** - Pydantic models for type-safe responses  
✅ **Conversation Threading** - MemorySaver for persistent state  
✅ **Gradio Web UI** - Beautiful interface with real-time updates  
✅ **Async/Await** - Non-blocking I/O for responsive UI  

### Files

```
📁 4_langgraph/
├── app.py                    (Gradio launcher)
├── sidekick.py              (Core agent logic)
├── sidekick_tools.py        (Tool definitions - includes Google Maps)
├── sandbox/                 (Working directory for file operations)
│   └── trip_map.html        (Generated trip maps)
└── pyproject.toml           (Dependencies)
```

**📂 [View Project 2 Files →](./4_langgraph/)**

### Quick Start

```bash
# Navigate to project
cd 4_langgraph

# Install dependencies (from root)
cd ..
uv sync
cd 4_langgraph

# Run the Sidekick
python app.py
```

### Environment Setup

Create a `.env` file with:
```
AUTOX_API_KEY=...
NTNET_USERNAME=...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
```

### Graph Architecture

```
START
  ↓
WORKER NODE (LLM with tools)
  ↓
  ├─ Has tool_calls? → YES → TOOLS NODE
  │                           ↓
  │                    Execute tool
  │                           ↓
  │                      Back to WORKER
  │
  └─ No tool_calls? → YES → EVALUATOR NODE
                              ↓
                      Success criteria met?
                              ↓
                         ├─ YES → END ✅
                         └─ NO → Back to WORKER (retry with feedback)
```

### 3 Key Scenarios

#### Scenario 1: Successful Task (Happy Path)
```
User: "Find Bitcoin price"
↓
Worker: Uses search tool
↓
Evaluator: Checks criteria met
↓
Returns: Price information ✅
```

#### Scenario 2: Needs Clarification
```
User: "Write a summary"
↓
Worker: Asks "Summary of what?"
↓
Evaluator: Needs user input
↓
Returns: Question to user ❓
```

#### Scenario 3: Failure & Retry (Most Complex)
```
User: "Create file with 100+ word summary"
↓
Worker: Creates file (20 words)
↓
Evaluator: REJECTS - too short, provides feedback
↓
Worker: Reads feedback, creates longer version (250 words)
↓
Evaluator: ACCEPTS ✅
```

### New: Trip Planning with Google Maps

The Sidekick now includes a **color-coded trip planner**:

```python
# Sidekick generates locations:
[
    {"name": "Hotel", "address": "Times Square, NYC", "color": "blue"},
    {"name": "Breakfast", "address": "Joe's Pizza, NYC", "color": "green"},
    {"name": "Museum", "address": "MoMA, NYC", "color": "red"}
]

# Tool creates: sandbox/trip_map.html
# Interactive map with markers, routes, and zoom
```

**Try it:** Ask Sidekick: "Plan a 5-stop NYC day trip and create a map"

### Understanding Super Steps

A **Super Step** = one iteration through the graph nodes:

```
Super Step 1: User → Worker → Router → Decision
Super Step 2: Tools execute → Worker → Router → Decision
Super Step 3: Evaluator checks → Decision (End or retry)
```

Each user interaction = separate super steps. The loop continues until success or user input needed.

### Async/Await Explained

The Sidekick uses `async/await` for non-blocking I/O:

```python
# Without async (BLOCKS UI):
result = graph.invoke(state)  # UI freezes for 10+ seconds

# With async (UI RESPONSIVE):
result = await graph.ainvoke(state)  # UI stays responsive
# Other coroutines can run while waiting
```

All on the same thread, but multiplexed execution!

---

## Project 3: Synthetic Dataset Generator

### What It Does

An intelligent LangGraph-based system that generates high-quality synthetic datasets for testing, training, and development purposes. The agent:

- **Generates realistic data** based on your use case description
- **Validates quality automatically** using AI-powered evaluation
- **Provides feedback loops** for iterative improvement
- **Allows manual editing** of specific records
- **Exports to JSON** with 50 customizable records
- **Ensures consistency** across data types and structure

### Key Features

✅ **AI-Powered Generation** - Uses GPT-4o to create contextually appropriate data  
✅ **Automatic Quality Evaluation** - Built-in evaluator checks 6 quality criteria:
   - Structure Consistency
   - Data Type Compliance
   - Realistic Values
   - Logical Coherence
   - Data Variety
   - Use Case Alignment

✅ **Iterative Improvement** - Regenerates data based on validation feedback  
✅ **Interactive CLI** - User-friendly command-line interface  
✅ **Manual Override** - Edit specific records or accept with quality warnings  
✅ **Structured Graph Flow** - LangGraph nodes for each stage  
✅ **JSON Export** - Clean, ready-to-use output format  

### Files

```
📁 4_langgraph/
└── DatasetGenerator.py  (Complete standalone application)
```

**📂 [View Project 3 File →](./4_langgraph/DatasetGenerator.py)**

### Quick Start

```bash
# Navigate to project
cd 4_langgraph

# Install dependencies (from root)
cd ..
uv sync
cd 4_langgraph

# Run the Dataset Generator
python DatasetGenerator.py
```

### Environment Setup

Uses the same `.env` configuration as Sidekick:
```
AUTOX_API_KEY=...
NTNET_USERNAME=...
```

### How It Works

```
User Input (Use Case + Example)
    ↓
GENERATE DATASET NODE (50 records)
    ↓
EVALUATE DATASET NODE
    ↓
    ├─ Quality Score ≥ 70? → YES → DISPLAY & EDIT
    │                                     ↓
    │                              User accepts? → EXPORT ✅
    │                                     ↓
    │                              User edits? → Back to DISPLAY
    │
    └─ Quality Score < 70? → NO → HANDLE FEEDBACK
                                        ↓
                                   Regenerate or Accept?
                                        ↓
                                   Back to GENERATE
```

### Graph Architecture

```
START
  ↓
INPUT COLLECTION NODE
  ↓
GENERATE DATASET NODE
  ↓
EVALUATE DATASET NODE
  ↓
  ├─ PASS (score ≥ 70) → DISPLAY & EDIT NODE
  │                           ↓
  │                      ├─ Accept → EXPORT NODE → END ✅
  │                      ├─ Regenerate → Back to GENERATE
  │                      ├─ Edit → Stay in DISPLAY
  │                      └─ Feedback → Back to GENERATE
  │
  └─ FAIL (score < 70) → HANDLE EVALUATION FEEDBACK NODE
                              ↓
                         ├─ Regenerate → Back to GENERATE
                         ├─ Accept anyway → DISPLAY & EDIT
                         └─ Restart → Back to INPUT COLLECTION
```

### Example Usage

**1. Provide Use Case:**
```
Describe your use case: Customer data for e-commerce platform
```

**2. Provide Example Structure:**
```
Provide example data format:
{
  "customer_id": "CUST-001",
  "name": "John Doe",
  "email": "john@example.com",
  "age": 35,
  "total_purchases": 1250.50,
  "membership_tier": "Gold"
}
```

**3. AI Generates 50 Records:**
```
Generated 50 synthetic records successfully.
```

**4. Automatic Validation:**
```
=== DATASET VALIDATION REPORT ===
Overall Score: 92/100
Recommendation: PASS

Compliance Checks:
- Structure Compliant: True
- Data Types Correct: True
- Realistic Data: True
- Logically Coherent: True
- Sufficient Variety: True
- Use Case Aligned: True
```

**5. Review and Export:**
```
Preview of generated data (first 3 records):
Record 1: {...}
Record 2: {...}
Record 3: {...}

Options:
1. Accept dataset and export
2. Regenerate dataset
3. Edit specific records
4. Provide feedback for improvement
```

**6. Dataset Saved:**
```
Dataset saved to 'synthetic_dataset.json'
```

### Quality Evaluation Criteria

The AI evaluator automatically checks:

| Criterion | Description |
|-----------|-------------|
| **Structure Consistency** | All records follow the example structure |
| **Data Type Compliance** | Values match expected types (string, number, date) |
| **Realistic Values** | Data is plausible and contextually appropriate |
| **Logical Coherence** | Related fields are logically consistent |
| **Data Variety** | Sufficient diversity, not repetitive |
| **Use Case Alignment** | Data fits the described use case |

**Passing Score:** 70/100 + "PASS" recommendation

### Regeneration & Feedback

The system supports multiple improvement paths:

**Automatic Regeneration:**
- Triggered when quality score < 70
- AI receives detailed validation report
- Addresses specific issues in next iteration

**User Feedback:**
- Provide custom instructions
- Example: "Make ages more diverse" or "Add international customers"
- AI incorporates feedback in regeneration

**Manual Editing:**
- Edit individual records by number
- Update specific fields with JSON input
- Useful for final tweaks

### Use Cases

Perfect for:
- **Testing**: Generate test data for applications
- **ML Training**: Create training datasets
- **API Mocking**: Populate mock API responses
- **Prototyping**: Quickly create realistic demo data
- **Database Seeding**: Generate initial database records
- **Documentation**: Create example data for docs

### Advanced Features

**State Management:**
```python
class DatasetGeneratorState(TypedDict):
    messages: list[BaseMessage]        # Conversation history
    use_case: str                      # User's use case
    example_data: str                  # Example structure
    generated_dataset: list[dict]      # Current dataset
    dataset_count: int                 # Number of records
    feedback: str                      # User feedback
    success_criteria_met: bool         # Quality check status
    validation_report: str             # Evaluation details
    retry_count: int                   # Generation attempts
```

**6 Graph Nodes:**
1. **Input Collection** - Gather use case and example
2. **Generate Dataset** - Create 50 synthetic records
3. **Evaluate Dataset** - AI quality assessment
4. **Handle Evaluation Feedback** - Decision on failed validation
5. **Display & Edit** - Preview, edit, or regenerate
6. **Export Dataset** - Save to JSON file

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- `uv` package manager
- API Keys:
  - Azure OpenAI / OpenAI
  - Pushover (for notifications)
  - Google Serper (for web search - optional, Sidekick works without it)

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/ChatWithMeAgent.git
cd ChatWithMeAgent

# Install dependencies
uv sync

# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### Corporate Proxy Configuration

If behind a corporate proxy (like Amdocs):

```python
# Already configured in sidekick_tools.py:
os.environ["NO_PROXY"] = ",".join([
    ".autox.corp.amdocs.azr",
    "chat.autox.corp.amdocs.azr",
    "localhost", "127.0.0.1"
])
```

**Key Rule:** Set environment variables BEFORE creating HTTP clients!

---

## Architecture Patterns

### Pattern 1: Tool Integration (Career Agent)

```
LLM + Tools → Tool Calls → Execution → Results → LLM Response
```

**Simple and Direct:** One pass through LLM with tools

### Pattern 2: Multi-Agent Loop (Sidekick)

```
Worker → Router → Tools/Evaluator → Decision → Worker or END
```

**Complex but Powerful:** Feedback loops, evaluators, retries

### Pattern 3: Iterative Generation with Validation (Dataset Generator)

```
Input → Generate → Evaluate → Feedback Loop → Regenerate or Accept → Export
```

**Quality-Focused:** Automatic evaluation with regeneration based on quality criteria

### Pattern 4: State Management

All projects use **LangChain State**:

```python
class State(TypedDict):
    messages: Annotated[List, add_messages]  # Reducer for concatenation
    other_fields: str                         # Simple overwrite
```

`add_messages` automatically appends new messages to history!

---

## Deployment

### Career Agent → HuggingFace Spaces

```bash
cd 1_foundations
uv run gradio deploy
# Follow prompts for app configuration
```

Result: Public URL like `huggingface.co/spaces/username/career_conversation`

### Sidekick → Cloud Platforms

**Option 1: Docker**
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN uv sync
CMD ["python", "4_langgraph/app.py"]
```

**Option 2: Railway/Render**
- Connect GitHub repo
- Set environment variables
- Deploy with one click

**Option 3: Local + Ngrok (for testing)**
```bash
python 4_langgraph/app.py
# In another terminal:
ngrok http 7860
```

### Dataset Generator → CLI/Script

**Local Execution:**
```bash
cd 4_langgraph
python DatasetGenerator.py
```

**Scheduled Generation (Cron/Task Scheduler):**
```bash
# Linux/Mac cron example
0 0 * * * cd /path/to/ChatWithMeAgent/4_langgraph && python DatasetGenerator.py
```

**Containerized:**
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN uv sync
CMD ["python", "4_langgraph/DatasetGenerator.py"]
```

---

## Key Technologies

| Technology | Purpose |
|-----------|---------|
| **LangChain** | LLM framework & tools |
| **LangGraph** | Graph-based agent orchestration |
| **Gradio** | Web UI for agents |
| **Playwright** | Web automation & scraping |
| **Folium** | Interactive map generation |
| **Geopy** | Address geocoding |
| **Azure OpenAI** | LLM with corporate proxy support |
| **Pydantic** | Structured outputs & validation |

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **504 Proxy Error** | Set `NO_PROXY` BEFORE creating HTTP clients |
| **Playwright NotImplementedError** | Comment out Windows event loop policy (see Cell 9, lab3) |
| **Nested Event Loop Error** | Use `nest_asyncio.apply()` in Jupyter |
| **Map not generating** | Ensure `sandbox/` folder exists and `geopy`/`folium` installed |
| **Pushover notifications not sending** | Verify `PUSHOVER_USER` and `PUSHOVER_TOKEN` in `.env` |
| **Dataset Generator JSON parse error** | AI output may include markdown formatting - check regex extraction in code |
| **Low quality scores repeatedly** | Provide more specific example data structure or detailed use case description |

---

## Learning Resources

- **Notes4.md** - Comprehensive documentation on:
  - LangGraph architecture & scenarios
  - Async/Await in Python
  - Proxy configuration
  - Google Maps tool implementation
  - Troubleshooting guide

- **Project Files:**
  - `1_foundations/ChatWithPushNotifications.ipynb` - Tool integration basics
  - `4_langgraph/app.py` - Gradio UI with async/await
  - `4_langgraph/sidekick.py` - Multi-agent orchestration
  - `4_langgraph/sidekick_tools.py` - Custom tool implementations
  - `4_langgraph/DatasetGenerator.py` - Complete iterative generation system

- **Notebooks:**
  - `4_langgraph/3_lab3.ipynb` - Browser automation intro
  - `4_langgraph/4_lab4.ipynb` - Multi-agent flows

---

## Contributing

Have improvements? Found bugs?

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

- **Issues:** Create an issue in this repository
- **Questions:** Open a discussion
- **LinkedIn:** [Connect with me](https://linkedin.com/in/eddonner/)
- **Email:** ed@edwarddonner.com

---

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [LangGraph](https://langgraph.dev/)
- UI by [Gradio](https://gradio.app/)
- Maps by [Folium](https://folium.readthedocs.io/)

**Happy coding! 🚀**