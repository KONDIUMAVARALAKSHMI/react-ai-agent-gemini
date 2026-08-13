# ReAct AI Agent (Gemini API)

A functional AI agent that solves multi-step problems by reasoning about
which tools to call, calling them, observing the results, and repeating
until it can give a final answer.

This project implements the **ReAct (Reasoning + Acting) pattern** using
Google Gemini's native function-calling capabilities.

The agent can:

- Decide which tool is required.
- Call tools sequentially.
- Receive tool results as observations.
- Use previous observations to decide the next action.
- Produce a final answer after completing the task.
- Stop safely after a configurable maximum number of steps.

---

## 1. Setup

### Requirements

- Python 3.9+
- Google Gemini API key
- Internet connection

> [!WARNING]
> The `requirements.txt` file in this repository lists `anthropic>=0.40.0`, but this project uses the Google Gemini Python SDK. You should install the actual dependencies manually:

```bash
pip install google-genai python-dotenv requests
```

### Create a virtual environment

Windows CMD:

```bash
python -m venv venv
venv\Scripts\activate
```

## 2. Configure the Gemini API Key

Create a `.env` file in the project root.

> [!WARNING]
> The `.env.example` file in this repository lists `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`. However, the actual Gemini implementation in `agent.py` requires `GEMINI_API_KEY` and `GEMINI_MODEL`.

Set the following variables in your `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Do not commit the real API key to GitHub. The `.env` file should remain private.

## 3. Running the Agent

### Interactive mode

Run:

```bash
python agent.py
```

The program will ask:

```
Enter your question for the agent:
```

Example input:

```
What's the weather in New York City, and what is 5 factorial?
```

### Command-line mode

You can also directly provide a question as arguments:

```bash
python agent.py "What's the weather in New York City, and what is 5 factorial?"
```

### Configure maximum steps

The agent supports a configurable maximum number of ReAct iterations via the `--max-steps` parameter.

Example:

```bash
python agent.py "What's the weather in New York City, and what is 5 factorial?" --max-steps 8
```

The default maximum is `8`. This prevents infinite execution.

## 4. Sample Multi-Tool Task

Use the following prompt to demonstrate the required multi-tool functionality:

`What's the weather in New York City, and what is 5 factorial?`

The agent should perform multiple tool calls.

A typical execution looks like:

```
USER QUERY: What's the weather in New York City, and what is 5 factorial?

======================================================================
STEP 1
======================================================================

[THOUGHT]
I need to find the weather in New York City and calculate 5 factorial. I'll start by retrieving the weather.

[ACTION] Calling tool: get_weather
         Arguments: {"city": "New York City"}

[OBSERVATION]
Weather in New York City: Sunny, 22C (72F), humidity 86%.

======================================================================
STEP 2
======================================================================

[THOUGHT]
Now I need to calculate 5 factorial. I'll use the calculate tool.

[ACTION] Calling tool: calculate
         Arguments: {"expression": "5!"}

[OBSERVATION]
Error evaluating expression '5!': invalid syntax

======================================================================
STEP 3
======================================================================

[THOUGHT]
The expression '5!' returned a syntax error. I will rewrite the expression as 5 * 4 * 3 * 2 * 1 to calculate 5 factorial.

[ACTION] Calling tool: calculate
         Arguments: {"expression": "5 * 4 * 3 * 2 * 1"}

[OBSERVATION]
Result of '5 * 4 * 3 * 2 * 1' = 120

======================================================================
STEP 4
======================================================================

[THOUGHT]
The weather in New York City is Sunny, 22C (72F), with 86% humidity. 5 factorial is 120.

[FINAL ANSWER]
The weather in New York City is Sunny, 22C (72F), with 86% humidity. 5 factorial is 120.
```

> [!NOTE]
> The weather data is fetched from a live API (`wttr.in`), so the exact weather values (temperature, condition, humidity) are dynamic and will vary depending on when and where the query is run.

The important part of the demonstration is that the agent:

1. Calls the weather tool.
2. Receives the weather observation.
3. Calls the calculation tool.
4. Handles the calculation error.
5. Tries a corrected calculation.
6. Receives the result.
7. Combines the observations into a final answer.

## 5. ReAct Architecture

The agent follows the ReAct pattern:

User Query
    |
    v
+----------------+
| Gemini LLM     |
+----------------+
    |
    v
  THOUGHT
    |
    v
  ACTION
    |
    v
+----------------+
| Python Tool    |
+----------------+
    |
    v
OBSERVATION
    |
    v
Conversation History
    |
    v
+----------------+
| Gemini LLM     |
+----------------+
    |
    v
Next Action OR Final Answer

The loop continues until Gemini provides a final answer or the maximum
number of steps is reached.

## 6. How the Agent Works

The main agent implementation is in:

agent.py

The agent performs the following operations.

Step 1 - Receive the user query

The user's question is added to the conversation history.

Example:

What's the weather in New York City, and what is 5 factorial?
Step 2 - Send the query to Gemini

The agent sends the conversation history and available tool definitions
to Gemini.

Step 3 - Determine the next action

Gemini can either:

Return a final text answer, or
Request a tool using function calling.
Step 4 - Execute the selected tool

The Python application finds the requested function and executes it.

Example:

get_weather("New York City")
Step 5 - Return the observation

The tool result is printed and added back to the conversation history.

Example (values are dynamic and fetched in real time):

```
[OBSERVATION]
Weather in New York City: Sunny, 22C (72F), humidity 86%.
```

Step 6 - Continue the loop

Gemini receives the observation and decides whether another tool is
required.

Step 7 - Produce the final answer

When no more tools are required, Gemini returns a final text response.

## 7. Tools Implemented

The project implements four distinct tools.

Tool	Type	Description
get_weather	Data Retrieval	Retrieves weather information for a city.
calculate	Calculation	Evaluates mathematical expressions.
write_file	File I/O	Writes text content to a file.
read_file	File I/O	Reads previously written file content.

Each tool:

Is implemented as a Python function.
Accepts arguments.
Returns a string result.
Can be selected by the Gemini model through function calling.
## 8. Tool Schemas

The available tools are described to Gemini using function declarations.

Each schema contains:

Tool name
Tool description
Parameters
Parameter types
Required fields

For example, the weather tool contains a city parameter:

get_weather
    |
    +-- city: string

This allows Gemini to determine how the tool should be called.

## 9. Multi-Step Reasoning Example

Consider:

What's the weather in New York City, and what is 5 factorial?

The agent can reason through the task as:

User Query
    |
    v
Need weather information
    |
    v
ACTION: get_weather
    |
    v
OBSERVATION: Weather information
    |
    v
Need factorial calculation
    |
    v
ACTION: calculate
    |
    v
OBSERVATION: 120
    |
    v
FINAL ANSWER

This demonstrates sequential use of multiple tools.

## 10. Error Handling

The agent handles tool errors without immediately crashing.

For example, if the calculation tool receives:

5!

and the expression evaluator does not support the ! syntax, the tool
returns an error observation.

Example:

[OBSERVATION]
Error evaluating expression '5!': invalid syntax

The observation is sent back to Gemini.

Gemini can then attempt a corrected expression such as:

5 * 4 * 3 * 2 * 1

which produces:

120

This demonstrates that tool errors become observations that the agent
can use in the next iteration.

## 11. Infinite Loop Protection

The agent contains a configurable max_steps parameter.

Default:

DEFAULT_MAX_STEPS = 8

The loop is implemented as:

for step in range(1, self.max_steps + 1):

If the agent cannot complete the task within the maximum number of
iterations, it stops gracefully and prints:

[STOPPED]
Max steps reached without completing the task.

This prevents an infinite execution loop.

## 12. Console Reasoning Trace

During execution, the agent displays the ReAct process in the console.

The trace contains:

- `STEP` header
- `[THOUGHT]` (when generated by the model)
- `[ACTION]` (when calling a tool, showing arguments)
- `[OBSERVATION]` (showing tool result)

Example:

```
======================================================================
STEP 1
======================================================================

[THOUGHT]
I will fetch the current weather conditions for New York City.

[ACTION] Calling tool: get_weather
         Arguments: {"city": "New York City"}

[OBSERVATION]
Weather in New York City: Sunny, 22C (72F), humidity 86%.
```

When the agent finishes:

```
[FINAL ANSWER]
The weather in New York City is Sunny, 22C (72F), with 86% humidity. 5 factorial is 120.
```

> [!NOTE]
> Weather values in the trace above are dynamic examples and will change depending on live API responses.

This makes the agent's tool-use process visible during execution.

## 13. Project Structure
react-agent/
│
├── agent.py
│   └── Gemini ReAct loop and application entry point
│
├── tools.py
│   └── Tool implementations and tool functions
│
├── agent_files/
│   └── Files created by the file tools
│
├── requirements.txt
│   └── Python dependencies
│
├── .env.example
│   └── Environment variable template
│
├── .env
│   └── Local Gemini API configuration
│
└── README.md
    └── Project documentation
## 14. Example Commands

**Basic calculation:**
```bash
python agent.py "What is 25 * 16?"
```

**Multi-tool task:**
```bash
python agent.py "What's the weather in New York City, and what is 5 factorial?"
```

**File operation task:**
```bash
python agent.py "Calculate 12 * 8, write the result to answer.txt, then read it back."
```

**Weather comparison:**
```bash
python agent.py "What's the weather in Tokyo and Paris, and which is warmer?"
```

**Custom maximum steps:**
```bash
python agent.py "What's the weather in New York City, and what is 5 factorial?" --max-steps 10
```

## 15. Requirements Checklist

- [x] At least 3 distinct functional tools
- [x] 4 tools implemented
- [x] Each tool accepts arguments and returns a string
- [x] JSON/function schema defined for every tool
- [x] Gemini function calling used for tool selection
- [x] ReAct-style iterative loop implemented
- [x] Tool observations added back to conversation history
- [x] Agent can call multiple tools sequentially
- [x] Multi-tool weather + calculation task demonstrated
- [x] Console trace displays THOUGHT / ACTION / OBSERVATION
- [x] Final answer produced after tool execution
- [x] Configurable max_steps parameter
- [x] Graceful stopping when maximum steps are reached
- [x] Tool errors are returned as observations
- [x] Environment variables used for API configuration

## 16. Technologies Used

- Python
- Google Gemini API
- Google GenAI Python SDK
- Function Calling
- ReAct Agent Pattern
- JSON Tool Schemas
- Python-dotenv
- REST/API-based weather retrieval
- File I/O
- Error Handling
## 17. Important Security Note

The Gemini API key is stored locally in `.env`. Do not commit `.env` to a public Git repository.

Add `.env` to `.gitignore`:

```
.env
venv/
__pycache__/
```

The `.env.example` file can be committed because it should contain only placeholder values.

## 18. Final Demonstration

Run:

```bash
python agent.py "What's the weather in New York City, and what is 5 factorial?"
```

The expected behavior flow is:

```
USER QUERY
    |
    v
Gemini (THOUGHT)
    |
    v
ACTION: get_weather
    |
    v
OBSERVATION
    |
    v
Gemini (THOUGHT)
    |
    v
ACTION: calculate
    |
    v
OBSERVATION
    |
    v
Gemini (THOUGHT)
    |
    v
FINAL ANSWER
```
