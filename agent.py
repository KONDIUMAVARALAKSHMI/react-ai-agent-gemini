"""
agent.py
A ReAct-pattern AI agent using Google's Gemini API.

The agent alternates between:
  1. THOUGHT      - the model explains its next step.
  2. ACTION       - the model calls a tool.
  3. OBSERVATION  - the tool result is fed back to the model.
"""

import os
import sys
import json
import argparse

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOL_FUNCTIONS


# Load environment variables from .env
load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_MAX_STEPS = 8


SYSTEM_PROMPT = (
    "You are a helpful AI agent that solves problems using the ReAct "
    "(Reasoning + Acting) pattern. "
    "For every turn, first briefly explain your next step in plain text "
    "as a thought. Then, if you need more information, call exactly one "
    "tool to take an action. "
    "Use the result returned by the tool as an observation. "
    "Then continue reasoning based on that observation. "
    "When you have enough information, provide a final answer. "
    "Do not call unnecessary tools."
)


class ReActAgent:
    """A minimal ReAct agent: Thought -> Action -> Observation."""

    def __init__(
        self,
        api_key: str = None,
        model: str = MODEL_NAME,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY")

        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=key)
        self.model = model
        self.max_steps = max_steps

    # =========================================================
    # Console trace helpers
    # =========================================================

    def _print_header(self, step: int):
        print(
            f"\n{'=' * 70}\n"
            f"STEP {step}\n"
            f"{'=' * 70}"
        )

    def _print_thought(self, thought: str):
        if thought.strip():
            print(f"\n[THOUGHT]\n{thought.strip()}")

    def _print_action(self, tool_name: str, tool_input: dict):
        print(f"\n[ACTION] Calling tool: {tool_name}")
        print(
            f"         Arguments: "
            f"{json.dumps(tool_input)}"
        )

    def _print_observation(self, observation: str):
        print(f"\n[OBSERVATION]\n{observation}")

    # =========================================================
    # Tool execution
    # =========================================================

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict
    ) -> str:

        func = TOOL_FUNCTIONS.get(tool_name)

        if func is None:
            return f"Error: unknown tool '{tool_name}'."

        try:
            return func(**tool_input)

        except Exception as exc:
            return (
                f"Error executing tool "
                f"'{tool_name}': {exc}"
            )

    # =========================================================
    # Main ReAct loop
    # =========================================================

    def run(self, user_query: str) -> str:

        print(f"\nUSER QUERY: {user_query}")

        # Initial user message
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=user_query)
                ],
            )
        ]

        # -----------------------------------------------------
        # Convert our tool schemas into Gemini tool declarations
        # -----------------------------------------------------

        gemini_tools = []

        for tool in self._get_tool_schemas():

            gemini_tools.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["input_schema"],
                )
            )

        tool_config = types.Tool(
            function_declarations=gemini_tools
        )

        # -----------------------------------------------------
        # ReAct loop
        # -----------------------------------------------------

        for step in range(
            1,
            self.max_steps + 1
        ):

            self._print_header(step)

            # -------------------------------------------------
            # Ask Gemini what to do next
            # -------------------------------------------------

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[tool_config],
                    temperature=0.2,
                ),
            )

            # -------------------------------------------------
            # Extract text and function calls
            # -------------------------------------------------

            thought_parts = []
            function_calls = []

            for candidate in response.candidates or []:

                if not candidate.content:
                    continue

                for part in candidate.content.parts or []:

                    # Text returned by Gemini
                    if part.text:
                        thought_parts.append(
                            part.text
                        )

                    # Function call returned by Gemini
                    if part.function_call:
                        function_calls.append(
                            part.function_call
                        )

            thought_text = "\n".join(
                thought_parts
            ).strip()

            # -------------------------------------------------
            # Print THOUGHT
            # -------------------------------------------------

            self._print_thought(
                thought_text
            )

            # -------------------------------------------------
            # No function call = final answer
            # -------------------------------------------------

            if not function_calls:

                final_answer = (
                    thought_text
                    or "(No final answer returned.)"
                )

                print(
                    f"\n[FINAL ANSWER]\n"
                    f"{final_answer}"
                )

                return final_answer

            # -------------------------------------------------
            # Add Gemini response to conversation history
            # -------------------------------------------------

            contents.append(
                response.candidates[0].content
            )

            # -------------------------------------------------
            # Execute tool calls
            # -------------------------------------------------

            tool_response_parts = []

            for call in function_calls:

                tool_name = call.name

                tool_input = dict(
                    call.args or {}
                )

                # Print ACTION
                self._print_action(
                    tool_name,
                    tool_input
                )

                # Execute tool
                observation = self._execute_tool(
                    tool_name,
                    tool_input
                )

                # Print OBSERVATION
                self._print_observation(
                    observation
                )

                # Prepare tool result for Gemini
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={
                            "result": observation
                        },
                    )
                )

            # -------------------------------------------------
            # Feed observations back to Gemini
            # -------------------------------------------------

            contents.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts,
                )
            )

        # -----------------------------------------------------
        # Max steps reached
        # -----------------------------------------------------

        message = (
            "Max steps reached without "
            "completing the task."
        )

        print(
            f"\n[STOPPED]\n{message}"
        )

        return message

    # =========================================================
    # Tool schemas
    # =========================================================

    @staticmethod
    def _get_tool_schemas():

        return [

            # -------------------------------------------------
            # Weather tool
            # -------------------------------------------------

            {
                "name": "get_weather",

                "description": (
                    "Retrieves the current weather "
                    "conditions (temperature, condition, "
                    "humidity) for a city."
                ),

                "input_schema": {
                    "type": "OBJECT",

                    "properties": {
                        "city": {
                            "type": "STRING",
                            "description": (
                                "The city name."
                            ),
                        }
                    },

                    "required": [
                        "city"
                    ],
                },
            },

            # -------------------------------------------------
            # Calculator tool
            # -------------------------------------------------

            {
                "name": "calculate",

                "description": (
                    "Evaluates a mathematical expression."
                ),

                "input_schema": {
                    "type": "OBJECT",

                    "properties": {
                        "expression": {
                            "type": "STRING",
                            "description": (
                                "The mathematical expression."
                            ),
                        }
                    },

                    "required": [
                        "expression"
                    ],
                },
            },

            # -------------------------------------------------
            # Write file tool
            # -------------------------------------------------

            {
                "name": "write_file",

                "description": (
                    "Writes text content to a file."
                ),

                "input_schema": {
                    "type": "OBJECT",

                    "properties": {

                        "filename": {
                            "type": "STRING",
                            "description": (
                                "The filename."
                            ),
                        },

                        "content": {
                            "type": "STRING",
                            "description": (
                                "The content to write."
                            ),
                        },
                    },

                    "required": [
                        "filename",
                        "content"
                    ],
                },
            },

            # -------------------------------------------------
            # Read file tool
            # -------------------------------------------------

            {
                "name": "read_file",

                "description": (
                    "Reads a previously written file."
                ),

                "input_schema": {
                    "type": "OBJECT",

                    "properties": {
                        "filename": {
                            "type": "STRING",
                            "description": (
                                "The filename."
                            ),
                        }
                    },

                    "required": [
                        "filename"
                    ],
                },
            },
        ]


# =============================================================
# Main
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the Gemini ReAct AI agent."
        )
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask the agent.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=(
            "Maximum number of ReAct steps."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=(
            "Gemini model name."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Check API key
    # ---------------------------------------------------------

    if not os.environ.get(
        "GEMINI_API_KEY"
    ):

        print(
            "ERROR: GEMINI_API_KEY is not set."
        )

        print(
            "Add your Gemini API key to .env"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Get user query
    # ---------------------------------------------------------

    query = (
        " ".join(args.query)
        if args.query
        else input(
            "Enter your question for the agent: "
        )
    )

    # ---------------------------------------------------------
    # Create agent
    # ---------------------------------------------------------

    agent = ReActAgent(
        model=args.model,
        max_steps=args.max_steps,
    )

    # ---------------------------------------------------------
    # Run agent
    # ---------------------------------------------------------

    agent.run(query)


if __name__ == "__main__":
    main()