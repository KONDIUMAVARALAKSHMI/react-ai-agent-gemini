"""
tools.py
Defines the tools available to the AI agent and their corresponding
JSON schemas for LLM function calling.
"""

import os
import re
import math
import ast
import operator

# Directory used by the file I/O tool. Kept sandboxed to this folder
# so the agent cannot read/write arbitrary paths on the machine.
FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_files")
os.makedirs(FILES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Tool 1: Weather lookup (data retrieval)
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """
    Retrieves the current weather for a given city using the free,
    no-API-key wttr.in service.
    """
    try:
        import requests
        response = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        condition = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        return (
            f"Weather in {city}: {condition}, {temp_c}C ({temp_f}F), "
            f"humidity {humidity}%."
        )
    except Exception as exc:
        return f"Error retrieving weather for '{city}': {exc}"


# ---------------------------------------------------------------------------
# Tool 2: Calculator (calculation)
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS = {
    "factorial": math.factorial,
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", None)
        if func_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Function '{func_name}' is not allowed.")
        args = [_safe_eval(a) for a in node.args]
        return _ALLOWED_FUNCS[func_name](*args)
    raise ValueError("Unsupported or unsafe expression.")


def calculate(expression: str) -> str:
    """
    Safely evaluates a mathematical expression. Supports +, -, *, /, //, %,
    **, parentheses, and functions like factorial(), sqrt(), sin(), cos(), etc.
    Does NOT use Python's built-in eval() for safety.
    """
    try:
        # Convenience: allow "5 factorial" style phrasing.
        match = re.fullmatch(r"\s*(\d+)\s*factorial\s*", expression, re.IGNORECASE)
        if match:
            expression = f"factorial({match.group(1)})"

        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"Result of '{expression}' = {result}"
    except Exception as exc:
        return f"Error evaluating expression '{expression}': {exc}"


# ---------------------------------------------------------------------------
# Tool 3: Write file (file I/O)
# ---------------------------------------------------------------------------
def write_file(filename: str, content: str) -> str:
    """
    Writes text content to a file inside the sandboxed agent_files directory.
    """
    try:
        safe_name = os.path.basename(filename)
        path = os.path.join(FILES_DIR, safe_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{safe_name}'."
    except Exception as exc:
        return f"Error writing file '{filename}': {exc}"


# ---------------------------------------------------------------------------
# Tool 4: Read file (file I/O)
# ---------------------------------------------------------------------------
def read_file(filename: str) -> str:
    """
    Reads and returns text content from a file inside the sandboxed
    agent_files directory.
    """
    try:
        safe_name = os.path.basename(filename)
        path = os.path.join(FILES_DIR, safe_name)
        if not os.path.exists(path):
            return f"Error: file '{safe_name}' does not exist."
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of '{safe_name}':\n{content}"
    except Exception as exc:
        return f"Error reading file '{filename}': {exc}"


# ---------------------------------------------------------------------------
# Tool registry: maps tool name -> callable
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "write_file": write_file,
    "read_file": read_file,
}


# ---------------------------------------------------------------------------
# JSON Schemas (Anthropic tool-use / function-calling format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": (
            "Retrieves the current weather conditions (temperature, "
            "condition, humidity) for a given city name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'New York City'.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluates a mathematical expression and returns the numeric "
            "result. Supports +, -, *, /, **, parentheses, and functions "
            "such as factorial(n), sqrt(n), sin(x), cos(x)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. 'factorial(5)' or '3*(4+2)'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "write_file",
        "description": "Writes the given text content to a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to write, e.g. 'notes.txt'.",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write into the file.",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Reads and returns the text content of a previously written file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to read, e.g. 'notes.txt'.",
                }
            },
            "required": ["filename"],
        },
    },
]
