# Agentic AI Frameworks

## Overview
Frameworks that help build agents. Complexity increases from left to right.

---

## 6 Frameworks (Low → High Complexity)

| # | Framework | Use Case | Complexity |
|---|-----------|----------|-----------|
| 1 | **No Framework** | Simple LLM calls, basic chains | ⭐ Low |
| 2 | **MCP** | Protocol for tool/resource discovery | ⭐ Low |
| 3 | **OpenAI Agents SDK** | Official, simple agent orchestration | ⭐⭐ Medium |
| 4 | **CrewAI** | Team-based multi-agent workflows | ⭐⭐⭐ Medium-High |
| 5 | **LangGraph** | State machines, complex workflows | ⭐⭐⭐⭐ High |
| 6 | **AutoGen** | Distributed agents, multi-turn chats | ⭐⭐⭐⭐⭐ Very High |

---

## Key Concepts

### Resources
Extra context added to prompts to improve answers.

**What:** Documents, data, code, documentation, URLs  
**How:** Retrieved based on relevance using RAG (Retrieval-Augmented Generation)  
**Why:** Prevents hallucination, grounds LLM in facts

**Example:**
```
Query: "What's the API response format?"
Resources: [API docs, code examples, past Q&A]
Prompt: "Use these resources to answer: [context] + [query]"
```

---

### Tools
Give LLMs the power to take actions (not just read).

**What:** Functions/APIs the LLM can call  
**Examples:** Query database, send email, fetch weather, run code  
**Benefit:** LLM becomes autonomous; can accomplish real work

**Tool Calling Flow:**

```
User: "What's the weather in NYC?"
    ↓
LLM thinks: "I need to call weather_tool"
    ↓
LLM requests: tool_call(name="weather", args={"city":"NYC"})
    ↓
Framework executes: actual_weather_data = weather_api("NYC")
    ↓
Result returned: "It's 72°F, sunny"
    ↓
LLM responds: "In NYC, it's 72°F and sunny"
```

**Code example:**
```python
from langchain.tools import tool

@tool
def query_database(sql: str):
    """Execute SQL query"""
    return db.execute(sql)

# LLM can now call: query_database(sql="SELECT * FROM users")
```

---

## Resources vs. Tools

| Resources | Tools |
|-----------|-------|
| Read-only context | Execute actions |
| Improve answer accuracy | Extend LLM capabilities |
| Passive (LLM reads) | Active (LLM calls) |
| Example: docs | Example: API calls |

---

## When to Use Each Framework

- **No Framework** – Simple chatbots, Q&A
- **MCP** – Standard protocol for tools/resources
- **OpenAI SDK** – Quick agent prototypes
- **CrewAI** – Multiple specialized agents (team)
- **LangGraph** – Complex workflows, state management
- **AutoGen** – Multi-agent conversations, distributed systems

---

## Typical Agent Stack

```
User Input
    ↓
[Router/Planner] (which agent/tool?)
    ↓
[LLM + Tools]
    ↓
[Execute Tools] (if needed)
    ↓
[Retrieve Resources] (context via RAG)
    ↓
[LLM Refines Answer]
    ↓
Output to User
```

---

## Agentic Design Patterns

### 1. **Reflection Pattern**
Agent evaluates its own responses before returning them.

```python
# Basic flow:
response = llm.invoke(prompt)
evaluation = evaluator_llm.invoke(f"Is this good? {response}")
if not evaluation.is_acceptable:
    response = llm.invoke(f"Improve this: {response}\nFeedback: {evaluation.feedback}")
```

**When to use:** Quality control, reducing hallucinations, professional outputs

---

### 2. **Tool Use Pattern**
Agent decides when to call external functions/APIs.

```python
llm_with_tools = llm.bind(tools=tools)
response = llm_with_tools.invoke(messages)

if response.tool_calls:
    # Execute tools and continue conversation
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
```

**When to use:** Database queries, API calls, file operations, notifications

---

### 3. **Planning Pattern**
Agent breaks complex tasks into steps before executing.

```python
# Step 1: Create plan
plan = planner_agent.invoke("Break this task into steps")

# Step 2: Execute each step
for step in plan.steps:
    result = executor_agent.invoke(step)
    
# Step 3: Synthesize results
final = synthesizer_agent.invoke(results)
```

**When to use:** Complex workflows, research tasks, multi-step processes

---

### 4. **Multi-Agent Pattern**
Multiple specialized agents collaborate.

```python
# Different agents for different tasks
researcher = Agent(role="researcher", tools=[search_tool])
writer = Agent(role="writer", tools=[write_tool])
critic = Agent(role="critic", tools=[])

# Agents work together
research = researcher.execute()
draft = writer.execute(research)
final = critic.execute(draft)
```

**When to use:** Complex domains, specialized expertise needed, team workflows

---

## Prompting Best Practices

### System Prompts
Define the agent's **role, behavior, and constraints**.

```python
system_prompt = """You are acting as {name}. 
You are answering questions on {name}'s website.

ROLE: Professional representative
TONE: Engaging, professional
CONSTRAINTS: Only answer using provided context
TOOLS: Use record_unknown_question when you don't know

Context:
{context}

With this context, please chat with the user."""
```

**Key elements:**
- Who/what the agent is
- Tone and style
- What it can/cannot do
- Available tools
- Context/resources

---

### User Prompts
The actual user input or task.

```python
user_prompt = f"""Here's the conversation:
{history}

Latest message: {message}

Please respond appropriately."""
```

---

## Structured Outputs with Pydantic

Force LLM to return data in a specific format.

```python
from pydantic import BaseModel

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str
    score: int

llm_structured = llm.with_structured_output(Evaluation)
result = llm_structured.invoke(messages)

# Now result is an Evaluation object:
print(result.is_acceptable)  # True/False
print(result.feedback)       # String
print(result.score)          # Integer
```

**Requirements:**
- API version must be `2024-08-01-preview` or later
- Works with `with_structured_output()`

---

## LangChain-Specific Patterns

### Message Types
```python
from langchain_core.messages import (
    SystemMessage,    # System instructions
    HumanMessage,     # User input
    AIMessage,        # LLM responses
    ToolMessage       # Tool results
)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What's 2+2?"),
    AIMessage(content="4"),
]
```

---

### Tool Binding
```python
# Define tools (OpenAI format)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        }
    }
}]

# Bind to LLM
llm_with_tools = llm.bind(tools=tools)

# Use it
response = llm_with_tools.invoke(messages)
if response.tool_calls:
    # Tool was called!
    tool_name = response.tool_calls[0]["name"]
    args = response.tool_calls[0]["args"]
```

---

## Common Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **SSL Certificate Errors** | Use `certifi.where()` for external APIs: `requests.post(url, verify=certifi.where())` |
| **Corporate Proxy Issues** | Set `NO_PROXY` env var: `os.environ["NO_PROXY"] = ".autox.corp.amdocs.azr"` |
| **Structured Output Errors** | Update API version to `2024-08-01-preview` or later |
| **Tool Not Called** | Improve tool description, make it more explicit |
| **Infinite Loops** | Add max iterations: `for i in range(max_iterations)` |
| **Token Limits** | Summarize history, use sliding window |

---

## UI/Deployment Tools

### Gradio
Quick UI for demos and testing.

```python
import gradio as gr

def chat(message, history):
    return llm.invoke(message).content

gr.ChatInterface(chat, type="messages").launch()
```

**Deployment:** `gradio deploy` to HuggingFace Spaces

---

### Streamlit
More customizable UI.

```python
import streamlit as st

st.title("My Agent")
user_input = st.text_input("Ask me anything")
if user_input:
    response = agent.run(user_input)
    st.write(response)
```

---

## Testing Your Agent

```python
# Unit tests for tools
def test_record_user_details():
    result = record_user_details(email="test@example.com", name="Test")
    assert result["recorded"] == "ok"

# Integration tests
def test_agent_flow():
    response = chat("What's your experience?", [])
    assert len(response) > 0
    assert "Vishnu" in response  # Check context is used
```

---

## Performance Tips

1. **Caching**: Cache expensive operations (embeddings, API calls)
2. **Async**: Use async for parallel tool calls
3. **Streaming**: Stream responses for better UX
4. **Model Selection**: Use smaller models for simple tasks
5. **Prompt Optimization**: Shorter, clearer prompts = faster responses

---

## Cost Optimization

| Strategy | Savings |
|----------|---------|
| Use GPT-4o-mini for simple tasks | 60-80% |
| Cache embeddings | 90%+ on repeated queries |
| Reduce context size | 50%+ |
| Batch API calls | 50% |
| Stream responses | Better UX, same cost |

---

## Security Considerations

- ✅ **Never** put API keys in code
- ✅ Use `.env` files and `.gitignore` them
- ✅ Validate user input before passing to LLM
- ✅ Rate limit API calls
- ✅ Sanitize LLM outputs before execution
- ✅ Use read-only tools when possible
- ❌ Don't let LLM execute arbitrary code
- ❌ Don't expose system prompts to users

---

## Quick Reference: LangChain vs OpenAI SDK

| Task | LangChain | OpenAI SDK |
|------|-----------|------------|
| **Basic call** | `llm.invoke(messages)` | `client.chat.completions.create(messages=...)` |
| **Response** | `response.content` | `response.choices[0].message.content` |
| **Tool binding** | `llm.bind(tools=...)` | Pass `tools=` to `create()` |
| **Tool calls** | `response.tool_calls` (list of dicts) | `response.choices[0].message.tool_calls` |
| **Messages** | `SystemMessage()`, `HumanMessage()` | `{"role": "system", "content": "..."}` |
| **Tool results** | `ToolMessage()` | `{"role": "tool", "content": "..."}` |