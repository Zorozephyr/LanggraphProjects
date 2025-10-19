# AI Agents & Workflow Patterns

## Setup
```bash
uv sync          # Create isolated environment
uv run <script>  # Run Python script
```

---

## AI Agents
**Definition:** Programs where LLM decisions drive the workflow. LLM calls tools, reads data, and decides next steps.

**Key Traits:**
- Multiple LLM calls
- Tool use (APIs, retrieval, code)
- Memory/state management
- Autonomy (LLM chooses actions)

---

## Workflows vs. Agents

| Workflow | Agent |
|----------|-------|
| Predefined code paths | LLM dynamically chooses steps |
| Low flexibility | High flexibility |
| Predictable, testable | Open-ended, novel problems |

---

## 5 Workflow Patterns

### 1. Prompt Chaining
Sequential LLM steps

```
Input → [Step 1] → [Step 2] → [Step 3] → Output
```
Use: predictable pipelines (extract → analyze → format)

---

### 2. Routing
Direct input to specialized handlers

```
              ┌→ [Handler A]
Input → [Router] ┼→ [Handler B]
              └→ [Handler C]
```
Use: different request types need different skills

---

### 3. Parallelization
Run independent tasks concurrently

```
           ┌→ [Worker 1] ┐
Input ─────┼→ [Worker 2] ┼→ [Merge] → Output
           └→ [Worker 3] ┘
```
Use: large workloads, multiple documents

---

### 4. Orchestrator–Worker
Central planner + dynamic workers

```
        [Orchestrator]
      ↙      ↓      ↘
 [W1]  [W2]  [W3]
      ↘      ↓      ↙
   [Aggregate] → Output
```
Use: complex tasks with variable steps

---

### 5. Evaluator–Optimizer
Iterate to improve quality

```
[Draft] → [Evaluate] ──feedback──→ [Refine] ──┐
  ↑                                             │
  └──────────────────iterate──────────────────┘
```
Use: quality-critical outputs (code, specs)

---

## Quick Decision Guide

| Scenario | Pattern |
|----------|---------|
| Simple, fixed steps | Prompt Chaining |
| Route by intent | Routing |
| Speed up wide tasks | Parallelization |
| Complex, variable | Orchestrator–Worker |
| High quality needed | Evaluator–Optimizer |

---

## Agent Risks & Guardrails

**Risks:**
- Unpredictable paths
- Unpredictable outputs
- Unpredictable costs
- Hard to monitor

**Guardrails:** Constraints ensure safe, consistent behavior within intended boundaries.

---

## Using Amdocs AutoX AI

### 1. Trust Certificates & Bypass Proxy
```python
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Trust corporate CA
os.environ["REQUESTS_CA_BUNDLE"] = r"C:\amdcerts.pem"
os.environ["SSL_CERT_FILE"]      = os.environ["REQUESTS_CA_BUNDLE"]

# Bypass proxy for AutoX
os.environ["NO_PROXY"] = ",".join(filter(None, [
    os.getenv("NO_PROXY", ""),
    ".autox.corp.amdocs.azr",
    "chat.autox.corp.amdocs.azr",
    "localhost", "127.0.0.1"
]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]
```

### 2. Sanity Check
```python
import requests

status = requests.get(
    "https://chat.autox.corp.amdocs.azr/api/v1/platforms/list",
    headers={"accept": "application/json"},
    proxies={"http": None, "https": None},
    timeout=30
).status_code
print(f"Status: {status}")  # Should print 200
```

### 3. Create LLM Client
```python
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

AUTOX_API_KEY  = os.getenv("AUTOX_API_KEY")
NTNET_USERNAME = (os.getenv("NTNET_USERNAME") or "").strip()

llm = AzureChatOpenAI(
    azure_endpoint="https://chat.autox.corp.amdocs.azr/api/v1/proxy",
    api_key=AUTOX_API_KEY,
    azure_deployment="gpt-4o-128k",
    model="gpt-4o-128k",
    temperature=0.1,
    openai_api_version="2023-05-15",
    default_headers={"username": NTNET_USERNAME, "application": "testing-proxyapi"},
)

resp = llm.invoke([HumanMessage(content="Hello!")])
print(resp.content)
```

### 4. Multi-Turn Conversation
```python
history = []

def ask(user_text):
    msgs = history + [HumanMessage(content=user_text)]
    ai = llm.invoke(msgs)
    history.extend([HumanMessage(content=user_text), AIMessage(content=ai.content)])
    return ai.content

# Use it
print(ask("Hi, I'm Nikhil"))
print(ask("What's my name?"))
```

---

## .env Requirements
```
AUTOX_API_KEY=<your_api_key>
NTNET_USERNAME=<your_username>
```