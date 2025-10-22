from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests

load_dotenv(override=True)


class Tools:
    def __init__(self):
        self.openai = OpenAI()

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

        self.tools = [{"type": "function", "function": self.record_user_details_json},
                      {"type": "function", "function": self.record_unknown_question_json}]

    def push(self, text):
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            }
        )

    def record_user_details(self, email, name="Name not provided", notes="not provided"):
        self.push(f"Recording {name} with email {email} and notes {notes}")
        return {"recorded": "ok"}

    def record_unknown_question(self, question):
        self.push(f"Recording {question}")
        return {"recorded": "ok"}

    def handle_tool_call(self, tool_calls):
        print("Handling tool calls:", tool_calls)
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
                print(msg)
                result = {"error": msg}

            results.append({"role": "tool", "content": json.dumps(
                result), "tool_call_id": tool_call.id})
        return results
