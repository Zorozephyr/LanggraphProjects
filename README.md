# ChatWithMeAgent: AI Agent Projects Suite

## Overview

This repository contains two advanced AI agent projects demonstrating different approaches to building intelligent conversational systems with tool integration, state management, and multi-agent patterns.

---

## 📋 Table of Contents

1. [Project 1: Career Conversation Agent (ChatWithPushNotifications)](#project-1-career-conversation-agent)
2. [Project 2: Sidekick Multi-Agent Assistant](#project-2-sidekick-multi-agent-assistant)
3. [Installation & Setup](#installation--setup)
4. [Architecture Patterns](#architecture-patterns)
5. [Deployment](#deployment)

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
📁 agents/1_foundations/
├── ChatWithPushNotifications.ipynb  (Main notebook)
├── app.py                           (Deployment script)
├── me/
│   ├── linkedin.pdf                 (Your LinkedIn profile)
│   └── summary.txt                  (Your professional summary)
└── requirements.txt
```

### Quick Start

```bash
# Navigate to project
cd agents/1_foundations

# Install dependencies
uv sync

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
cd agents/1_foundations
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
📁 agents/4_langgraph/
├── app.py                    (Gradio launcher)
├── sidekick.py              (Core agent logic)
├── sidekick_tools.py        (Tool definitions - includes Google Maps)
├── sandbox/                 (Working directory for file operations)
│   └── trip_map.html        (Generated trip maps)
└── pyproject.toml           (Dependencies)
```

### Quick Start

```bash
# Navigate to project
cd agents/4_langgraph

# Install dependencies
cd ../..
uv sync
cd agents/4_langgraph

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

### Pattern 3: State Management

Both projects use **LangChain State**:

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
cd agents/1_foundations
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
CMD ["python", "agents/4_langgraph/app.py"]
```

**Option 2: Railway/Render**
- Connect GitHub repo
- Set environment variables
- Deploy with one click

**Option 3: Local + Ngrok (for testing)**
```bash
python agents/4_langgraph/app.py
# In another terminal:
ngrok http 7860
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

---

## Learning Resources

- **Notes4.md** - Comprehensive documentation on:
  - LangGraph architecture & scenarios
  - Async/Await in Python
  - Proxy configuration
  - Google Maps tool implementation
  - Troubleshooting guide

- **Notebooks:**
  - `agents/4_langgraph/3_lab3.ipynb` - Browser automation intro
  - `agents/4_langgraph/4_lab4.ipynb` - Multi-agent flows
  - `agents/1_foundations/ChatWithPushNotifications.ipynb` - Tool integration

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