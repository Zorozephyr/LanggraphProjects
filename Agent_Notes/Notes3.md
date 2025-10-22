# LangGraph - State Machine Workflows

## The LangChain Ecosystem

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  LangChain   │──────│  LangGraph   │──────│  LangSmith   │
│              │      │              │      │              │
│ LLM chains   │      │ State graphs │      │ Monitoring   │
│ Prompts      │      │ Workflows    │      │ Debugging    │
│ Tools        │      │ Agents       │      │ Traces       │
└──────────────┘      └──────────────┘      └──────────────┘
```

**LangChain**: Basic building blocks (LLMs, prompts, tools)  
**LangGraph**: Complex workflows as state machines  
**LangSmith**: Observability and debugging

---

## Core Concepts

### Graphs
Agent workflows represented as directed graphs.

```
     START
       │
       ↓
   ┌────────┐
   │ Node A │ ──→ Decision
   └────────┘         │
                 ┌────┴────┐
                 ↓         ↓
            ┌────────┐ ┌────────┐
            │ Node B │ │ Node C │
            └────────┘ └────────┘
                 │         │
                 └────┬────┘
                      ↓
                     END
```

---

### State
**Shared snapshot** of the application at any point in time.

```python
from typing import Annotated
from langgraph.graph import MessagesState

class State(MessagesState):
    counter: int = 0
    user_input: str = ""
    results: list[str] = []
```

**Key properties:**
- Passed to every node
- Immutable (nodes return new state)
- Updated via reducers

---

### Nodes
**Python functions** that implement logic.

```python
def my_node(state: State) -> State:
    """Receives current state, returns updated state"""
    return {
        "counter": state["counter"] + 1,
        "results": state["results"] + ["new result"]
    }
```

**Visual:**
```
     ┌─────────────────────┐
     │      Node           │
Input│  ┌──────────────┐   │Output
State│─→│ Python Logic │──→│State
     │  └──────────────┘   │
     └─────────────────────┘
```

---

### Edges
**Connections** between nodes (determine flow).

**Types:**

1. **Fixed Edge**: Always go to next node
```python
graph.add_edge("node_a", "node_b")
```

2. **Conditional Edge**: Dynamic routing
```python
def router(state: State) -> str:
    if state["counter"] > 5:
        return "node_b"
    return "node_c"

graph.add_conditional_edges("node_a", router)
```

**Visual:**
```
Fixed:              Conditional:
A → B              A ──→ condition ──→ B
                            ↓
                            C
```

---

## 5 Steps to Create a Graph

### 1. Define State Class
```python
from typing import Annotated
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    count: int = 0
    user_name: str = ""
```

### 2. Initialize Graph Builder
```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)
```

### 3. Add Nodes
```python
def process_input(state: AgentState) -> AgentState:
    return {"count": state["count"] + 1}

graph.add_node("process", process_input)
```

### 4. Add Edges
```python
# Fixed edge
graph.add_edge("process", "next_node")

# Conditional edge
graph.add_conditional_edges("process", router_function)

# Set entry point
graph.set_entry_point("process")
graph.set_finish_point("next_node")
```

### 5. Compile the Graph
```python
app = graph.compile()

# Run it
result = app.invoke({"count": 0, "user_name": "Alice"})
```

---

## State Immutability

**Concept**: Nodes don't modify state directly; they return new state.

```python
# ❌ Wrong - mutating state
def bad_node(state: State) -> State:
    state["counter"] += 1  # Direct mutation
    return state

# ✅ Correct - returning new state
def good_node(state: State) -> State:
    return {"counter": state["counter"] + 1}
```

**Why?**
- Enables concurrent execution
- Makes state changes trackable
- Prevents race conditions

**Flow:**
```
Old State → Node Function → New State
    ↓                           ↓
{count: 5}                  {count: 6}
           (immutable)
```

---

## Reducers

**Special functions** that merge new state with existing state.

### Built-in Reducers

#### 1. **Default (Replace)**
```python
class State(TypedDict):
    name: str  # New value replaces old
```

#### 2. **Add to List**
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list[str], add]  # Appends to list
```

**Example:**
```python
# Initial state
state = {"messages": ["hello"]}

# Node returns
return {"messages": ["world"]}

# Result (merged)
{"messages": ["hello", "world"]}  # Not replaced!
```

#### 3. **Custom Reducer**
```python
def merge_dicts(existing: dict, new: dict) -> dict:
    return {**existing, **new}

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

---

## Understanding `Annotated`

Python's type hint for adding metadata to types.

### Syntax
```python
from typing import Annotated

# Format: Annotated[type, reducer_function]
field: Annotated[list, add]
```

### Examples

**Without reducer (default behavior):**
```python
class State(TypedDict):
    name: str  # Replaces old value
```

**With reducer (custom behavior):**
```python
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]  # Appends instead of replacing
```

**Visual:**
```
Without Annotated:
Old: ["a"]  →  New: ["b"]  →  Result: ["b"]  (replaced)

With Annotated[list, add]:
Old: ["a"]  →  New: ["b"]  →  Result: ["a", "b"]  (merged)
```

---

## Common Patterns

### Pattern 1: Simple Linear Flow
```python
graph.add_node("start", start_node)
graph.add_node("process", process_node)
graph.add_node("end", end_node)

graph.add_edge("start", "process")
graph.add_edge("process", "end")
graph.set_entry_point("start")
graph.set_finish_point("end")
```

**Visual:**
```
START → Process → END
```

---

### Pattern 2: Conditional Routing
```python
def route(state: State) -> str:
    if state["score"] > 0.8:
        return "approve"
    return "review"

graph.add_conditional_edges("evaluate", route, {
    "approve": "finish",
    "review": "human_review"
})
```

**Visual:**
```
            ┌──→ Approve → Finish
Evaluate ───┤
            └──→ Review → Human
```

---

### Pattern 3: Parallel Execution
```python
from operator import add

class State(TypedDict):
    results: Annotated[list, add]

# Multiple nodes can run concurrently
graph.add_node("task_a", task_a)
graph.add_node("task_b", task_b)
graph.add_node("merge", merge_results)

graph.add_edge("start", "task_a")
graph.add_edge("start", "task_b")
graph.add_edge("task_a", "merge")
graph.add_edge("task_b", "merge")
```

**Visual:**
```
       ┌──→ Task A ──┐
Start ─┤             ├──→ Merge
       └──→ Task B ──┘
```

---

## MessagesState (Built-in)

Pre-built state for chat applications.

```python
from langgraph.graph import MessagesState

class MyState(MessagesState):
    # Already includes:
    # messages: Annotated[list[BaseMessage], add]
    
    # Add your fields:
    user_name: str = ""
    context: dict = {}
```

**Benefits:**
- Automatic message history management
- Built-in reducer for appending messages
- Compatible with LangChain message types

---

## Complete Example

```python
from typing import Annotated
from operator import add
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define State
class AgentState(MessagesState):
    counter: Annotated[int, lambda x, y: x + y] = 0
    results: Annotated[list, add] = []

# 2. Create nodes
def process(state: AgentState):
    return {
        "counter": 1,
        "messages": [AIMessage(content="Processing...")],
        "results": ["step1"]
    }

def finalize(state: AgentState):
    return {
        "messages": [AIMessage(content=f"Done! Counter: {state['counter']}")],
        "results": ["complete"]
    }

# 3. Build graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_node("finalize", finalize)
graph.add_edge("process", "finalize")
graph.set_entry_point("process")
graph.set_finish_point("finalize")

# 4. Compile & run
app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="Start")],
    "counter": 0,
    "results": []
})

# Result:
# {
#   "counter": 1,
#   "messages": [HumanMessage("Start"), AIMessage("Processing..."), AIMessage("Done! Counter: 1")],
#   "results": ["step1", "complete"]
# }
```

---

## Key Advantages of LangGraph

✅ **Visual workflows** - Easy to understand complex logic  
✅ **State management** - Centralized, trackable  
✅ **Parallelism** - Run nodes concurrently  
✅ **Conditional logic** - Dynamic routing  
✅ **Debuggable** - Clear execution path  
✅ **Composable** - Reusable nodes and graphs

---

## When to Use LangGraph

| Scenario | Use LangGraph? |
|----------|----------------|
| Simple chatbot | ❌ Use plain LangChain |
| Multi-step workflow | ✅ Yes |
| Conditional branching | ✅ Yes |
| Parallel task execution | ✅ Yes |
| Human-in-the-loop | ✅ Yes |
| Complex state management | ✅ Yes |

---

## Quick Reference

```python
# Import
from langgraph.graph import StateGraph, MessagesState
from typing import Annotated
from operator import add

# State with reducer
class State(MessagesState):
    items: Annotated[list, add]

# Graph
graph = StateGraph(State)
graph.add_node("name", function)
graph.add_edge("from", "to")
graph.add_conditional_edges("node", router_fn)
graph.set_entry_point("start")
graph.set_finish_point("end")
app = graph.compile()

# Run
result = app.invoke(initial_state)
```





