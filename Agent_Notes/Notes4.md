# Langgraph - Sidekick Application: Complete Scenarios Guide

## Core Concept: Super Step

**Super Step** can be considered a single iteration over the graph nodes. 
- Nodes that run in parallel are part of the same super step
- Nodes that run sequentially belong to separate super steps
- Every user interaction is a separate super step

---

## Sidekick Application Overview

The Sidekick is a multi-agent AI assistant that:
1. Takes user requests and success criteria
2. Uses tools to complete tasks (browser automation, file management, Python execution, web search)
3. Evaluates its own work
4. Iterates until success or asks for user input
5. Maintains conversation memory per session (thread_id)

### Available Tools:
- **Playwright Browser Tools**: navigate, extract text, click elements, get links
- **File Management**: read, write, create, delete files in `sandbox/` directory
- **Python REPL**: Execute Python code with print output
- **Web Search**: Google Serper API for online searches
- **Wikipedia**: Query Wikipedia for information
- **Push Notifications**: Send alerts to user
- **Push Notifications**: Send alerts to user

---

## SCENARIO 1: Successful Task Completion (Happy Path)

### Situation:
User provides a clear task with specific success criteria that the agent can complete in one iteration.

### Example:
```
User Input: "Find the current Bitcoin price"
Success Criteria: "Must show current price in USD"
```

### Super Step Breakdown:

#### Super Step 1: Worker Node (1st invocation)
```
Worker reads state:
- messages: [HumanMessage("Find the current Bitcoin price")]
- success_criteria: "Must show current price in USD"

System Message: "You are a helpful assistant... Success criteria: Must show current price in USD"

LLM thinks:
"I need to find Bitcoin's current price. I can use the search tool."

Returns:
AIMessage(
    content="",
    tool_calls=[
        {"name": "search", "args": {"query": "current bitcoin price USD"}, "id": "call_123"}
    ]
)

Router Decision: "tools" → Has tool_calls
```

State After:
```python
messages = [
    HumanMessage("Find the current Bitcoin price"),
    AIMessage(tool_calls=[search])
]
```

#### Super Step 2: Tools Node
```
Execute search tool:
- Query: "current bitcoin price USD"
- Returns: "Bitcoin is trading at $42,500 USD"

Create ToolMessage:
ToolMessage(
    content="Bitcoin is trading at $42,500 USD",
    tool_call_id="call_123",
    name="search"
)
```

State After:
```python
messages = [
    HumanMessage("Find the current Bitcoin price"),
    AIMessage(tool_calls=[search]),
    ToolMessage(content="Bitcoin is trading at $42,500 USD", ...)
]
```

#### Super Step 3: Worker Node (2nd invocation)
```
Worker sees:
- All previous messages including tool result
- Success criteria: "Must show current price in USD"

LLM thinks:
"I have the Bitcoin price: $42,500 USD. This meets the criteria. No need for more tools."

Returns:
AIMessage(
    content="The current Bitcoin price is $42,500 USD.",
    tool_calls=[]  # ← EMPTY!
)

Router Decision: "evaluator" → No tool_calls
```

State After:
```python
messages = [
    HumanMessage("Find the current Bitcoin price"),
    AIMessage(tool_calls=[search]),
    ToolMessage(content="Bitcoin is trading at $42,500 USD", ...),
    AIMessage(content="The current Bitcoin price is $42,500 USD.", tool_calls=[])
]
```

#### Super Step 4: Evaluator Node
```
Evaluator receives:
- Last response: "The current Bitcoin price is $42,500 USD."
- Success criteria: "Must show current price in USD"
- Conversation history: [all messages]

LLM evaluates:
"Does this show current price in USD? YES!
Does user input needed? NO!"

Returns:
EvaluatorOutput(
    feedback="Successfully provided current Bitcoin price in USD",
    success_criteria_met=True,
    user_input_needed=False
)

Router Decision: "END" → success_criteria_met=True
```

Final State:
```python
messages = [
    HumanMessage("Find the current Bitcoin price"),
    AIMessage(tool_calls=[search]),
    ToolMessage(content="Bitcoin is trading at $42,500 USD", ...),
    AIMessage(content="The current Bitcoin price is $42,500 USD."),
    AIMessage(content="Evaluator Feedback: Successfully provided current Bitcoin price in USD")
]
success_criteria_met = True
user_input_needed = False
```

### UI Display:
```
User: Find the current Bitcoin price
Agent: The current Bitcoin price is $42,500 USD.
Evaluator: Successfully provided current Bitcoin price in USD
```

### Graph Flow:
```
START → worker (1st) → router → tools → worker (2nd) → router → evaluator → END ✅
         (4 super steps total)
```

---

## SCENARIO 2: Agent Asks for Clarification (User Input Needed)

### Situation:
User provides a vague task where the agent needs clarification before proceeding.

### Example:
```
User Input: "Write me a summary"
Success Criteria: "Write a summary"
```

### Super Step Breakdown:

#### Super Step 1: Worker Node (1st invocation)
```
Worker reads state:
- messages: [HumanMessage("Write me a summary")]
- success_criteria: "Write a summary"

LLM thinks:
"The user wants a summary, but they didn't specify:
- Summary of WHAT? (a website? a document? a topic?)
- How long? (1 paragraph? 1 page?)
- What format? (bullet points? prose?)

I need to ask for clarification before proceeding."

Returns:
AIMessage(
    content="Question: What would you like me to summarize? Please provide either a URL, topic, or text to summarize. Also, how long would you like the summary to be?",
    tool_calls=[]  # ← NO tools
)

Router Decision: "evaluator" → No tool_calls
```

#### Super Step 2: Evaluator Node
```
Evaluator receives:
- Last response: "Question: What would you like me to summarize?..."
- Success criteria: "Write a summary"

LLM evaluates:
"Has the assistant completed the task? NO
Does the assistant have a valid question? YES, they're asking for clarification
Should I ask user for input? YES"

Returns:
EvaluatorOutput(
    feedback="Assistant correctly identified ambiguity and asked for clarification. User input needed to specify: (1) What to summarize, (2) Length, (3) Format",
    success_criteria_met=False,
    user_input_needed=True  # ← TRUE!
)

Router Decision: "END" → user_input_needed=True
```

### UI Display:
```
User: Write me a summary
Agent: Question: What would you like me to summarize? Please provide either a URL, topic, or text to summarize. Also, how long would you like the summary to be?
Evaluator: Assistant correctly identified ambiguity and asked for clarification. User input needed...
```

### What Happens Next:
```
User can now:
1. Press Reset → Start fresh conversation
2. Type new message in success_criteria → Update criteria and press Go!
3. Type new message in textbox → Provide clarification without reset

If user provides clarification WITHOUT reset:
- Same thread_id used
- New message appended to history
- Agent continues with context of previous question
```

---

## SCENARIO 3: Task Fails Evaluation & Agent Retries (FINAL SCENARIO - MOST COMPLEX)

### Situation:
Agent completes the task but evaluator rejects it. Agent receives feedback and RETRIES to fix it.

### Example:
```
User Input: "Create a file called summary.txt with a summary of Python"
Success Criteria: "File must be created with Python summary containing at least 100 words"
```

### Super Step Breakdown:

#### Super Steps 1-4: First Attempt (Agent Creates File)
```
Worker:
- Uses file_write tool to create summary.txt
- Writes basic Python summary: "Python is a programming language..."

Evaluator checks:
- Does file exist? YES
- Is summary at least 100 words? NO (only 20 words)

Evaluator returns:
feedback="The file was created but the summary is too short. Must contain at least 100 words about Python."
success_criteria_met=False
user_input_needed=False  # ← Agent can retry!

Router: "worker" → Loop back to worker!
```

State After First Attempt:
```python
messages = [
    HumanMessage("Create a file called summary.txt with a summary of Python"),
    AIMessage(tool_calls=[file_write]),
    ToolMessage(content="File created successfully"),
    AIMessage(content="I've created summary.txt with a Python summary"),
    AIMessage(content="Evaluator Feedback: File created but summary too short. Need 100+ words")
]
feedback_on_work = "The file was created but the summary is too short..."
success_criteria_met = False
user_input_needed = False
```

#### Super Step 5: Worker Node (2nd invocation) - WITH FEEDBACK

```python
def worker(state: State) -> Dict[str, Any]:
    system_message = f"""You are a helpful assistant...
    Success criteria: File must be created with Python summary containing at least 100 words
    
    Previously you thought you completed the assignment, but your reply was rejected!
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    
    → "The file was created but the summary is too short..."
    
    With this feedback, please continue the assignment, ensuring that you meet 
    the success criteria or have a question for the user.
    """
    
    # LLM now sees:
    # 1. All previous messages (including failed attempt)
    # 2. Success criteria (need 100+ words)
    # 3. EXPLICIT FEEDBACK on why it failed
    
    LLM thinks:
    "Ah! The evaluator rejected my previous attempt because:
    - File exists (good!)
    - But summary is too short
    
    I need to write a MUCH LONGER summary about Python (100+ words minimum)
    Let me rewrite/append to the file with a comprehensive Python summary"
    
    Returns:
    AIMessage(
        content="",
        tool_calls=[
            {"name": "file_write", 
             "args": {"file_path": "summary.txt", 
                     "content": "Python is a high-level, interpreted programming language...
                                 [200+ words comprehensive summary]"}, 
             "id": "call_456"}
        ]
    )

    Router Decision: "tools" → Has tool_calls
```

State After:
```python
messages = [
    HumanMessage("Create a file called summary.txt with a summary of Python"),
    AIMessage(tool_calls=[file_write]),                    # 1st attempt
    ToolMessage(content="File created successfully"),      # 1st attempt
    AIMessage(content="I've created summary.txt..."),      # 1st attempt
    AIMessage(content="Evaluator Feedback: Too short"),    # Evaluator rejected
    AIMessage(tool_calls=[file_write]),                    # 2nd attempt - NEW with longer content!
]
```

#### Super Step 6: Tools Node (2nd execution)
```
Execute file_write tool:
- Overwrites summary.txt with 200+ word Python summary
- Returns: "File successfully written with 250 words"

ToolMessage:
content="File summary.txt successfully written with comprehensive Python summary (250 words)"
```

State After:
```python
messages = [
    ... (5 previous messages) ...,
    AIMessage(tool_calls=[file_write]),
    ToolMessage(content="File successfully written with 250 words")  # ← NEW
]
```

#### Super Step 7: Worker Node (3rd invocation)
```
Worker sees:
- All previous attempts including feedback
- Tool executed successfully with longer content
- Success criteria: 100+ words

LLM thinks:
"Good! The file has been written with 250 words. That meets the criteria (100+ words).
No need for more tools, let me provide final answer."

Returns:
AIMessage(
    content="I've successfully created summary.txt with a comprehensive Python summary containing 250 words, meeting the 100+ word requirement.",
    tool_calls=[]  # ← EMPTY, done with tools
)

Router Decision: "evaluator" → No tool_calls
```

#### Super Step 8: Evaluator Node (2nd evaluation)
```
Evaluator receives:
- Last response: "I've successfully created summary.txt..."
- Success criteria: "File must contain at least 100 words"
- Previous feedback: "Summary was too short"
- Full conversation history: Shows first attempt AND retry

LLM evaluates:
"Has agent addressed the feedback? YES
- First attempt: 20 words → REJECTED
- Agent received feedback
- Second attempt: 250 words → ACCEPTED
- Criteria met? YES

success_criteria_met=True
user_input_needed=False"

Returns:
EvaluatorOutput(
    feedback="Task successful! Agent created summary.txt with 250 words about Python. 
              Agent correctly responded to feedback and improved the solution to meet criteria.",
    success_criteria_met=True,
    user_input_needed=False
)

Router Decision: "END" → success_criteria_met=True ✅
```

### Final State:
```python
state = {
    "messages": [
        HumanMessage("Create a file called summary.txt with a summary of Python"),
        AIMessage(tool_calls=[file_write]),           # 1st attempt
        ToolMessage("File created"),                  # 1st attempt
        AIMessage("I've created summary.txt"),        # 1st attempt
        AIMessage("Evaluator: Too short"),            # Evaluator rejected
        AIMessage(tool_calls=[file_write]),           # 2nd attempt - IMPROVED!
        ToolMessage("File written 250 words"),        # 2nd attempt success
        AIMessage("I've successfully created..."),    # 2nd attempt response
        AIMessage("Evaluator: Task successful!")      # ← FINAL SUCCESS!
    ],
    "success_criteria": "File must contain at least 100 words",
    "feedback_on_work": "The file was created but the summary is too short...",
    "success_criteria_met": True,   # ← FINALLY TRUE!
    "user_input_needed": False
}
```

### UI Display:
```
User: Create a file called summary.txt with a summary of Python
Agent: I've created summary.txt with a Python summary
Evaluator: File created but summary too short. Need 100+ words

[Agent retries internally]

Agent: I've successfully created summary.txt with comprehensive Python summary (250 words)
Evaluator: Task successful! Agent created summary.txt with 250 words...
```

### Complete Flow:
```
START 
  ↓
worker (1st: creates short file)
  ↓
tools (writes file)
  ↓
worker (2nd: proposes to user)
  ↓
evaluator (REJECTS - too short, feedback given)
  ↓
worker (3rd: receives feedback, creates longer file) ← KEY DIFFERENCE FROM SCENARIO 1!
  ↓
tools (writes better file)
  ↓
worker (4th: proposes solution)
  ↓
evaluator (ACCEPTS - success!)
  ↓
END ✅

Total: 8 super steps (vs 4 for successful single attempt)
```

---

## Key Differences Between Scenarios

| Aspect | Scenario 1: Success | Scenario 2: Clarification | Scenario 3: Failure & Retry |
|--------|-------------------|------------------------|---------------------------|
| **Super Steps** | 4 | 2 | 8+ |
| **Evaluator Decision** | ✅ Accepted immediately | ❓ User input needed | ❌ Rejected → feedback → retry |
| **Agent Gets Feedback?** | No | No | YES! Via `feedback_on_work` |
| **System Message** | Standard | Standard | **INCLUDES FAILURE FEEDBACK** |
| **Router After Evaluator** | END | END | **"worker"** (retry!) |
| **Conversation History** | Grows normally | Short (just question) | **LONG** (shows full retry loop) |
| **User Interaction** | Receives final answer | Must clarify request | **Receives improved answer** |

---

## Critical Flow Element: Feedback Loop (Scenario 3 Only)

### The `feedback_on_work` Variable:

**First Evaluation (Failed):**
```python
# Evaluator sets:
feedback_on_work = "The file was created but the summary is too short..."
user_input_needed = False
success_criteria_met = False

# Router returns "worker" → RETRY!
```

**Worker Receives Feedback:**
```python
if state.get("feedback_on_work"):
    system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected!
    Here is the feedback: {state["feedback_on_work"]}
    
    With this feedback, please continue the assignment...
    """
```

**Agent Uses Feedback to Improve:**
```python
# LLM now knows:
# 1. What was attempted before (from messages history)
# 2. Why it failed (from feedback_on_work)
# 3. What needs fixing (from success_criteria)

# Agent can now make a BETTER attempt!
```

---

## Thread & Memory Persistence

### With Same Thread ID (NO RESET):
```
Both tasks share history:
Super Step 1: "Find Bitcoin price" → SUCCESS
Super Step 2: "Create Python summary" → Conversation sees Bitcoin messages too
```

### With Different Thread ID (RESET):
```
reset() creates:
- New Sidekick instance
- New thread_id
- New MemorySaver entry
- Fresh conversation history
```

---

## Summary

- **Scenario 1 (Happy Path)**: User → Worker → Tools → Worker → Evaluator → SUCCESS
- **Scenario 2 (Clarification)**: User → Worker (asks question) → Evaluator (needs input) → END
- **Scenario 3 (Failure & Retry)**: User → Worker → Tools → Evaluator (REJECT) → Worker (retry with feedback) → Tools → Worker → Evaluator (ACCEPT) → SUCCESS

**The key innovation**: Scenario 3 shows how agents can improve through feedback loops without human intervention! 

---

# Python Async/Await: Complete Explanation

## What is `await`?

`await` is a keyword that tells Python: **"This operation will take time. Pause HERE and let other tasks run while we wait for this to complete."**

**Key Point**: `await` does NOT stop all threads. It **pauses ONLY the current coroutine** and allows other coroutines to run.

---

## Single-Threaded vs Async

### ❌ WRONG Understanding:
```
"await blocks everything until it's done"
```

### ✅ CORRECT Understanding:
```
"await pauses THIS coroutine and lets OTHER coroutines run on the SAME thread"
```

**Python is still single-threaded with async**, but it multiplexes (switches between) many coroutines.

---

## Example 1: WITHOUT await (Blocking - Bad)

```python
import time

def download_file(url):
    print(f"Downloading {url}...")
    time.sleep(5)  # ← BLOCKS for 5 seconds, nothing else can run!
    print(f"Downloaded {url}")
    return "file content"

# Sequential execution (total: 15 seconds)
print("Start:", time.time())
file1 = download_file("url1")  # 5 seconds
file2 = download_file("url2")  # 5 seconds  
file3 = download_file("url3")  # 5 seconds
print("End:", time.time())

# Output:
# Start: 1000
# Downloading url1...
# Downloaded url1
# Downloading url2...
# Downloaded url2
# Downloading url3...
# Downloaded url3
# End: 1015
```

**The problem**: Each download BLOCKS. Total time = 5+5+5 = 15 seconds. Other code can't run!

---

## Example 2: WITH async/await (Non-Blocking - Good)

```python
import asyncio

async def download_file_async(url):
    print(f"Downloading {url}...")
    await asyncio.sleep(5)  # ← PAUSES this coroutine, other coroutines can run!
    print(f"Downloaded {url}")
    return "file content"

async def main():
    print("Start:", time.time())
    
    # Create all tasks (don't await yet)
    task1 = download_file_async("url1")
    task2 = download_file_async("url2")
    task3 = download_file_async("url3")
    
    # Run all concurrently
    results = await asyncio.gather(task1, task2, task3)
    
    print("End:", time.time())
    return results

asyncio.run(main())

# Output:
# Start: 1000
# Downloading url1...
# Downloading url2...
# Downloading url3...
# Downloaded url1
# Downloaded url2
# Downloaded url3
# End: 1005
```

**The benefit**: ALL THREE download in PARALLEL (on same thread!). Total time = 5 seconds. **10x faster!**

---

## How Execution Flows with await

### Single Line-by-Line:

```python
async def task_a():
    print("A: start")
    await asyncio.sleep(2)
    print("A: end")

async def task_b():
    print("B: start")
    await asyncio.sleep(1)
    print("B: end")

async def main():
    # Run both concurrently
    await asyncio.gather(task_a(), task_b())

asyncio.run(main())
```

### Execution Timeline:

```
Time 0.0s:
  │ main() starts
  │ task_a() starts
  │ task_b() starts
  │
  ├─ A: start          ← Prints
  ├─ await sleep(2)    ← Pauses task_a, frees up thread
  │
  ├─ B: start          ← NOW task_b runs while task_a sleeps
  ├─ await sleep(1)    ← Pauses task_b, frees up thread
  │
  └─ Both waiting... (thread is idle, could run other tasks)

Time 1.0s:
  │ task_b's sleep completes
  │
  ├─ B: end            ← Prints (1 second elapsed)
  │ (task_b finishes)
  │
  └─ task_a still sleeping...

Time 2.0s:
  │ task_a's sleep completes
  │
  ├─ A: end            ← Prints (2 seconds elapsed)
  │ (task_a finishes)
  │
  └─ Both done!

Total time: 2 seconds (not 3!)
```

**What happened:**
- When task_a hit `await`, it PAUSED
- Thread switched to task_b
- When task_b hit `await`, it PAUSED
- Both slept concurrently
- No CPU wasted waiting

---

## Comparison: await vs Regular Function

### Regular Function (No await):
```python
def sync_download(url):
    print(f"Downloading {url}...")
    time.sleep(2)  # ← BLOCKS EVERYTHING
    print(f"Downloaded {url}")
    return "content"

result = sync_download("url")  # Waits 2 seconds, blocks completely
```

**Execution:**
```
┌─────────────────────────────┐
│ Main Thread: BLOCKED        │
│ (waiting for sleep to end)  │
│ Can't do anything else!     │
└─────────────────────────────┘
       ↓ 2 seconds ↓
```

### Async Function (With await):
```python
async def async_download(url):
    print(f"Downloading {url}...")
    await asyncio.sleep(2)  # ← PAUSES this coroutine only
    print(f"Downloaded {url}")
    return "content"

await async_download("url")  # Yields control, other coroutines can run
```

**Execution:**
```
┌─────────────────────────────┐
│ Coroutine A: PAUSED         │
│ Thread can run Coroutine B! │
│ Thread can run Coroutine C! │
└─────────────────────────────┘
       ↓ 2 seconds ↓
    (but other tasks ran!)
```

---

## Key Insight: The Event Loop

Python's async system uses an **Event Loop**:

```python
# Simplified version of what asyncio.run() does:

event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

# Keeps running...
while True:
    # 1. Check which coroutines are waiting
    waiting_coroutines = [task_a, task_b, task_c]
    
    # 2. Run the coroutines that are ready (not awaiting)
    for coro in waiting_coroutines:
        if coro.is_ready():
            run_until_await(coro)  # Run until next await
    
    # 3. If nothing is ready, sleep briefly
    
    # 4. When an await completes, mark that coroutine as ready
    
    # 5. Loop back to step 1
```

**The loop switches between coroutines, not threads!**

---

## Real-World Analogy: Restaurant Waiters

### WITHOUT async (Blocking - 1 waiter):
```
Waiter takes order from Table 1
Waiter waits at kitchen while food cooks (5 mins) ← BLOCKS
Waiter brings food to Table 1
Waiter takes order from Table 2
Waiter waits at kitchen while food cooks (5 mins) ← BLOCKS
...

Result: Only 1 table served per 5 minutes (inefficient!)
```

### WITH async (Coroutines - 1 waiter, multiple tables):
```
Waiter takes order from Table 1
Waiter PAUSES (await) while kitchen cooks for Table 1
  ↓ Waiter now serves Table 2!
Waiter takes order from Table 2
Waiter PAUSES (await) while kitchen cooks for Table 2
  ↓ Waiter now serves Table 3!
Waiter takes order from Table 3
Waiter PAUSES (await) while kitchen cooks for Table 3
Table 1's food is ready! Waiter brings it
Table 2's food is ready! Waiter brings it
Table 3's food is ready! Waiter brings it

Result: 3 tables served in ~5 minutes (efficient!)
```

**One waiter (single thread) serving multiple tables (coroutines)!**

---

## Applied to Sidekick Code

### In app.py:
```python
async def process_message(sidekick, message, success_criteria, history):
    results = await sidekick.run_superstep(message, success_criteria, history)
    return results, sidekick
```

**What happens:**
```
1. process_message starts
2. Calls: await sidekick.run_superstep(...)
3. PAUSES here while run_superstep executes
   (Other Gradio UI events can be processed!)
4. When run_superstep finishes, resume
5. Return results
```

### In sidekick.py:
```python
async def run_superstep(self, message, success_criteria, history):
    config = {"configurable": {"thread_id": self.sidekick_id}}
    
    state = {...}
    
    result = await self.graph.ainvoke(state, config=config)
    # ↑ PAUSES here while graph runs
    # (Gradio UI remains responsive!)
    
    return history + [user, reply, feedback]
```

**What happens:**
```
1. run_superstep starts
2. Calls: await self.graph.ainvoke(...)
3. PAUSES while graph processes:
   - Worker node
   - Tools node
   - Evaluator node
   (Can take 10+ seconds)
4. During these 10 seconds:
   - Gradio UI is NOT frozen
   - User can interact with other elements
   - Other coroutines can run
5. When graph finishes, resume
6. Return result
```

**Without async:**
```python
result = self.graph.invoke(state, config=config)  # ← BLOCKS for 10 seconds
# UI would FREEZE for entire 10 seconds! ❌
```

---

## Does await Stop Other Threads?

### Answer: NO! 

```python
import asyncio
import threading

print(f"Main thread: {threading.current_thread().name}")

async def coroutine_a():
    print(f"A: {threading.current_thread().name}")  # Same thread!
    await asyncio.sleep(2)
    print("A: done")

async def coroutine_b():
    print(f"B: {threading.current_thread().name}")  # Same thread!
    await asyncio.sleep(1)
    print("B: done")

async def main():
    await asyncio.gather(coroutine_a(), coroutine_b())

asyncio.run(main())

# Output:
# Main thread: MainThread
# A: MainThread
# B: MainThread
# B: done
# A: done
```

**All on SAME thread!** `await` doesn't create threads. It just pauses coroutines on the same thread.

---

## Summary Table

| Aspect | `time.sleep()` | `await asyncio.sleep()` |
|--------|---|---|
| **Blocks** | Entire thread | Only current coroutine |
| **Other tasks** | Can't run | Can run |
| **Concurrency** | ❌ No | ✅ Yes |
| **Threads used** | 1 thread | 1 thread |
| **Use case** | Blocking operations | I/O operations |
| **Total time for 3×2sec tasks** | 6 seconds | 2 seconds |

---

## Quick Reference: When to Use await

```python
# ✅ USE await for:
await asyncio.sleep(2)           # Waiting
await http_client.get(url)        # Network I/O
await file.read()                 # File I/O
await graph.ainvoke(state)        # LLM calls
await database.query()            # Database queries

# ❌ DON'T USE await for:
time.sleep(2)                     # Use asyncio.sleep instead
json.loads(data)                  # Instant, no I/O
calculations                      # Instant, no waiting
```

The rule: **await for anything that takes time due to I/O, not CPU work!** 

---

# Sidekick Proxy Error: Root Cause & Fix

## Error:
```
httpcore.ProxyError: 504 Unknown Host
```

## Root Cause

The error occurred in `sidekick_tools.py` in the `llmConnection()` function. The problem was **ordering of operations**:

### ❌ WRONG ORDER (Original Code):

```python
def llmConnection()->AzureChatOpenAI:
    # Step 1: Create httpx clients FIRST
    http_client = httpx.Client(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )
    async_http_client = httpx.AsyncClient(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )
    
    # Step 2: Set NO_PROXY AFTER ❌ (TOO LATE!)
    os.environ["NO_PROXY"] = ",".join(filter(None, [
        os.getenv("NO_PROXY",""),
        ".autox.corp.amdocs.azr",
        "chat.autox.corp.amdocs.azr",
        "localhost","127.0.0.1"
    ]))
    
    # Step 3: Create Azure LLM with already-created clients
    return AzureChatOpenAI(
        ...,
        http_client=http_client,
        http_async_client=async_http_client
    )
```

**Why this fails:**
- httpx clients read environment variables AT CREATION TIME
- Setting `NO_PROXY` AFTER creating clients = clients ignore the setting ❌
- Clients still try to route requests through corporate proxy
- Proxy can't resolve OpenAI endpoint → `504 Unknown Host` error

---

## ✅ CORRECT ORDER (Fixed Code):

Move proxy configuration to **MODULE LEVEL**, before ANY clients are created:

```python
load_dotenv(override=True)

# 🔥 SET PROXY BYPASS FIRST - AT MODULE LEVEL
os.environ["NO_PROXY"] = ",".join(filter(None, [
    os.getenv("NO_PROXY",""),
    ".autox.corp.amdocs.azr",
    "chat.autox.corp.amdocs.azr",
    "localhost","127.0.0.1"
]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

# Now safe to use variables
pushover_token = os.getenv("PUSHOVER_TOKEN")
...

def llmConnection()->AzureChatOpenAI:
    # NOW create httpx clients - AFTER NO_PROXY is set ✅
    http_client = httpx.Client(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )
    
    async_http_client = httpx.AsyncClient(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )
    
    # Return Azure LLM with properly configured clients
    return AzureChatOpenAI(
        ...,
        http_client=http_client,
        http_async_client=async_http_client
    )
```

**Why this works:**
- Environment variables set at module load time
- httpx clients created AFTER variables are set
- Clients read and respect `NO_PROXY` setting ✅
- Requests bypass proxy correctly ✅
- Can reach OpenAI endpoint successfully ✅

---

## Key Lesson: Environment Variables & Client Initialization

### Rule:
**Set environment variables BEFORE creating clients that depend on them!**

```python
# ❌ WRONG:
client = create_http_client()
os.environ["SETTING"] = "value"  # Too late!

# ✅ RIGHT:
os.environ["SETTING"] = "value"  # First!
client = create_http_client()    # Then create
```

### Timeline Comparison:

```
WRONG ORDER:
┌─────────────────────────────────────┐
│ 1. Create httpx client              │ ← Reads env vars (NO_PROXY not set!)
│ 2. Set os.environ["NO_PROXY"]       │ ← Too late, client already created
│ 3. Create Azure LLM                 │ ← Uses already-created client
│ Result: ❌ Proxy error              │
└─────────────────────────────────────┘

RIGHT ORDER:
┌─────────────────────────────────────┐
│ 1. Set os.environ["NO_PROXY"]       │ ← First!
│ 2. Create httpx client              │ ← Reads env vars (sees bypass!)
│ 3. Create Azure LLM                 │ ← Uses properly-configured client
│ Result: ✅ Works correctly          │
└─────────────────────────────────────┘
```

---

## Verification

After fix, test with:
```python
import os
print("NO_PROXY:", os.environ.get("NO_PROXY"))

from sidekick_tools import llmConnection
llm = llmConnection()
# Should connect to Azure successfully without proxy errors
```

---

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| **504 Unknown Host** | NO_PROXY set after client creation | Set NO_PROXY at module level |
| **Proxy not bypassed** | httpx didn't read env var | Create clients after env vars |
| **Corporate cert errors** | Client config order | Move setup to top of file |
| **Multiple LLM instances** | Called `llmConnection()` twice | Same function, reused config |

This is a common pattern in configuration management: **Always establish environment/configuration first, then initialize clients that depend on it!** 

---

# Google Maps Trip Planning Tool

## Overview

The Sidekick now includes a **Google Maps trip planning tool** that:
- ✅ Creates interactive color-coded maps
- ✅ Marks locations with custom colors
- ✅ Draws routes between stops
- ✅ Helps visualize trip itineraries
- ✅ NO API KEY REQUIRED (uses free OpenStreetMap)

---

## Installation

### Step 1: Install Required Libraries

```bash
pip install folium geopy
```

**What these do:**
- `folium`: Creates interactive maps using Leaflet.js
- `geopy`: Converts addresses to coordinates (geocoding)

### Step 2: Verify Installation

```python
import folium
import geopy
print("✅ folium and geopy installed!")
```

---

## How It Works

### Tool Function

```python
def create_trip_map(locations_json: str, map_title: str = "Trip Planning Map") -> str:
    """
    Create a color-coded interactive map for trip planning.
    
    Input: JSON array of locations
    Output: Interactive HTML map saved to sandbox/trip_map.html
    """
```

### Location Format

The AI provides locations as a JSON array:

```json
[
    {
        "name": "Hotel Arrival",
        "address": "Times Square Hotel, New York",
        "color": "blue"
    },
    {
        "name": "Breakfast",
        "address": "Joe's Pizza, New York",
        "color": "green"
    },
    {
        "name": "Statue of Liberty",
        "address": "Liberty Island, New York",
        "color": "red"
    },
    {
        "name": "Central Park",
        "address": "Central Park, New York",
        "color": "orange"
    },
    {
        "name": "Dinner",
        "address": "Tavern on the Green, New York",
        "color": "purple"
    }
]
```

### Available Colors

```
blue, red, green, purple, orange, brown, pink, gray, yellow
```

---

## Usage Examples

### Example 1: Plan a New York City Day Trip

**User Input:**
```
Message: "Plan a day trip to New York City with 5 stops"
Success Criteria: "Create a color-coded map showing hotel, breakfast, statue of liberty, central park, and dinner locations"
```

**What Sidekick Does:**

1. **Worker Node:**
   - Understands the task: NYC day trip with 5 stops
   - Decides to use create_trip_map tool
   - Identifies locations and assigns colors

2. **Tools Node:**
   - Calls create_trip_map with:
     ```json
     [
       {"name": "Hotel", "address": "Times Square, NYC", "color": "blue"},
       {"name": "Breakfast", "address": "Joe's Pizza, NYC", "color": "green"},
       {"name": "Statue of Liberty", "address": "Liberty Island", "color": "red"},
       {"name": "Central Park", "address": "Central Park, NYC", "color": "orange"},
       {"name": "Dinner", "address": "Tavern on the Green, NYC", "color": "purple"}
     ]
     ```

3. **Map Creation:**
   - Geocodes each address (converts to lat/lon)
   - Creates folium map with markers
   - Draws connecting lines between stops
   - Color-codes by stop type
   - Saves to `sandbox/trip_map.html`

4. **Result:**
   ```
   ✅ Trip map created successfully! Saved to sandbox/trip_map.html
   📍 Locations marked: 5
   🗺️ Open the HTML file in a browser to view the interactive map.
   ```

### Example 2: Europe Multi-City Tour

**User Input:**
```
Message: "Plan a 3-city Europe tour: London, Paris, Amsterdam"
Success Criteria: "Map should show hotels, attractions, restaurants for each city with different colors for each city"
```

**Map Output:**
- Blue markers: London (hotel, big ben, restaurant)
- Red markers: Paris (hotel, eiffel tower, restaurant)
- Green markers: Amsterdam (hotel, canal tour, restaurant)

---

## Map Features

### Interactive Elements

When you open `trip_map.html` in a browser:

✅ **Click markers** → See location details
✅ **Hover markers** → See location names
✅ **Zoom/Pan** → Navigate the map
✅ **Route lines** → See connections between stops
✅ **Color coding** → Visual organization by stop type

### Example Map Output:

```
┌─────────────────────────────────────┐
│  🗺️  NYC Day Trip Map               │
│                                     │
│     [🔵] Hotel (Stop 1)            │
│     ↓ (connecting line)            │
│     [🟢] Breakfast (Stop 2)        │
│     ↓                              │
│     [🔴] Statue of Liberty (Stop 3) │
│     ↓                              │
│     [🟠] Central Park (Stop 4)     │
│     ↓                              │
│     [🟣] Dinner (Stop 5)           │
│                                     │
│ Zoom: 12  OpenStreetMap            │
└─────────────────────────────────────┘
```

---

## How to View the Map

### After Trip Planning:

1. **Sidekick outputs:**
   ```
   ✅ Trip map created successfully! 
   Saved to sandbox/trip_map.html
   ```

2. **Open in browser:**
   ```
   📂 Navigate to: C:\AgenticAI\agents\4_langgraph\sandbox\trip_map.html
   🖱️ Double-click to open in default browser
   Or: Right-click → Open with → Chrome/Firefox
   ```

3. **Interact with map:**
   - Zoom with scroll wheel
   - Pan by dragging
   - Click markers for details
   - Hover for tooltips

---

## Integration in Sidekick

### Graph Flow with Maps Tool

```
User: "Plan a trip to Japan"
  ↓
Worker Node: Decides to use create_trip_map
  ↓
Tools Node: Creates map with:
  - Tokyo hotels (blue)
  - Kyoto temples (red)
  - Osaka restaurants (green)
  ↓
Map saved: sandbox/trip_map.html
  ↓
Evaluator: Success! Trip planned with map
  ↓
User sees: Trip description + map location message
```

---

## Advanced: Color-Coding Strategy

### By Stop Type:

```python
{
    "blue": "Hotels/Accommodations",
    "red": "Major attractions",
    "green": "Dining/Restaurants",
    "orange": "Transportation hubs",
    "purple": "Museums/Cultural sites",
    "brown": "Nature/Parks",
    "pink": "Shopping/Stores",
    "gray": "Other stops"
}
```

### Example Trip:

```json
[
    {"name": "Airport", "address": "LAX, Los Angeles", "color": "orange"},
    {"name": "Hotel Check-in", "address": "Beverly Hills, LA", "color": "blue"},
    {"name": "Griffith Observatory", "address": "Griffith Observatory, LA", "color": "red"},
    {"name": "Dinner", "address": "Sunset Boulevard, LA", "color": "green"},
    {"name": "Beach Day", "address": "Santa Monica Beach, LA", "color": "brown"},
    {"name": "Shopping", "address": "The Grove, LA", "color": "pink"}
]
```

---

## Error Handling

### Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Could not geocode address" | Address not found | Use more specific address |
| Empty map | No locations provided | Ensure JSON is valid |
| JSON parsing error | Invalid JSON format | Check brackets, quotes |
| File not found | sandbox folder missing | Create `sandbox/` folder |

### Example Valid Address Formats:

✅ `"Times Square, New York, USA"`
✅ `"40.758, -73.985"` (coordinates)
✅ `"123 Main Street, New York, NY 10001"`
✅ `"Eiffel Tower, Paris, France"`

---

## Usage in Sidekick Chat

### Prompt Example:

```
Message: "I'm planning a 5-day trip to Tokyo. 
Create a detailed itinerary with a map showing 
all major attractions, restaurants, and hotels"

Success Criteria: "Map should include:
- 3 hotels (blue markers)
- 10 attractions (red markers)  
- 5 restaurants (green markers)
- Color-coded by type
- Interactive map saved as HTML"
```

### What Happens:

1. **Sidekick researches** Tokyo attractions
2. **Creates detailed itinerary** with locations
3. **Calls create_trip_map** with all locations
4. **Generates map** with color-coded markers
5. **Returns:** Itinerary + map file path

---

## Technical Details

### Map Features Used

```python
folium.Map()           # Base map with OpenStreetMap tiles
folium.Marker()        # Location markers with colors
folium.Icon()          # Custom colored icons
folium.Popup()         # Click-to-see details
folium.PolyLine()      # Route connections
fit_bounds()           # Auto-zoom to all markers
```

### Geocoding

```python
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="sidekick_trip_planner")
location = geolocator.geocode("Times Square, NYC")
# Returns: Latitude, Longitude, Address details
```

**Free & No API Key Required!** Uses OpenStreetMap data.

---

## Next Steps

### Enhancement Ideas:

1. **Distance calculation**: Show distances between stops
2. **Estimated time**: Add travel time estimates
3. **Elevation profile**: Show terrain for hiking trips
4. **Weather data**: Show weather for each stop
5. **Cost breakdown**: Add budget info to markers
6. **Photo gallery**: Embed photos of attractions
7. **Export options**: Save as PDF, image, or print

### Try It Now:

```
Ask Sidekick: "Plan a road trip from Los Angeles to San Francisco 
with 5 stops along the way. Create a color-coded map."
```

The tool is now integrated and ready to help with trip planning! 🗺️🎉 