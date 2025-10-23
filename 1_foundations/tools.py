from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests

# Load environment variables from .env file (overriding existing ones)
load_dotenv(override=True)

class Tools:
    # Set as class variables for consistency and efficiency
    PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
    PUSHOVER_USER = os.getenv("PUSHOVER_USER")

    def __init__(self):
        # Initialize OpenAI API client
        self.openai = OpenAI()

        # JSON schema for recording user details
        self.record_user_details_json = {
            "name": "record_user_details",
            "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email address of this user"
                    },
                    "name": {
                        "type": "string",
                        "description": "The user's name, if they provided it"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any additional information about the conversation that's worth recording to give context"
                    }
                },
                "required": ["email"],
                "additionalProperties": False
            }
        }

        # JSON schema for recording an unknown question
        self.record_unknown_question_json = {
            "name": "record_unknown_question",
            "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question that couldn't be answered"
                    },
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }

        # List of tool metadata for OpenAI function-calling
        self.tools = [
            {"type": "function", "function": self.record_user_details_json},
            {"type": "function", "function": self.record_unknown_question_json},
        ]

    def push(self, text: str) -> None:
        """Send a notification via Pushover."""
        if not self.PUSHOVER_TOKEN or not self.PUSHOVER_USER:
            print("Pushover credentials missing")
            return
        try:
            resp = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": self.PUSHOVER_TOKEN,
                    "user": self.PUSHOVER_USER,
                    "message": text,
                }
            )
            if not resp.ok:
                print(f"Pushover request failed: {resp.text}")
        except Exception as e:
            print(f"Pushover network error: {e}")

    def record_user_details(self, email: str, name: str = None, notes: str = None) -> dict:
        """Record user details and push a notification."""
        nm = name if name else "<Name not provided>"
        nt = notes if notes else "<No notes provided>"
        self.push(f"Recording {nm} with email {email} and notes {nt}")
        return {"recorded": "ok"}

    def record_unknown_question(self, question: str) -> dict:
        """Record an unknown question and push a notification."""
        self.push(f"Recording {question}")
        return {"recorded": "ok"}

    # Handle an array of tool calls (as used by OpenAI function-calling API)
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Look up the tool as an instance method first
            tool = getattr(self, tool_name, None)
            if callable(tool):
                try:
                    result = tool(**arguments)
                except Exception as e:
                    # Return an error dict which is JSON serializable
                    result = {"error": str(e)}
            else:
                # Tool not found on this instance; return a JSON-serializable message
                msg = f"Tool {tool_name} not found"
                result = {"error": msg}

            # Package tool result for OpenAI API
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

