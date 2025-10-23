# Import required libraries for environment, AI, file handling, and UI
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel
from evaluator import Evaluator
from tools import Tools

# Load environment variables from .env file
load_dotenv(override=True)


class Me:
    # Pydantic model for evaluation results
    class Evaluation(BaseModel):
        is_acceptable: bool
        feedback: str

    def __init__(self):
        # Initialize OpenAI client for GPT models
        self.openai = OpenAI()
        # Initialize Gemini client using OpenAI-compatible API
        self.gemini = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.name = "Don Kukral"
        # Extract text from LinkedIn PDF
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        # Load personal summary from text file
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

        # Combine personal information for context
        self.about = {'name': self.name,
                      'summary': self.summary, 'linkedin': self.linkedin}
        # Initialize evaluator and tools
        self.evaluator = Evaluator()
        self.tools = Tools()

    def system_prompt(self):
        # Create system prompt for AI to act as on behalf of the loaded user
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        # Add personal context to the prompt
        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def _normalize_content(self, content):
        """Convert content to a JSON-serializable string."""
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content)
        except Exception:
            return str(content)

    def _build_tool_calls_list(self, tool_calls):
        """Build a serializable list of tool calls."""
        tc_list = []
        for tc in tool_calls:
            tc_dict = {
                "id": getattr(tc, "id", None),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                },
            }
            tc_list.append(tc_dict)
        return tc_list

    def _serialize_assistant_message(self, assistant_message, tool_calls):
        """Convert assistant message to a serializable dict."""
        role = getattr(assistant_message, "role", "assistant")
        content = self._normalize_content(getattr(assistant_message, "content", None))
        
        if not tool_calls:
            return {"role": role, "content": content}
        
        # Try different serialization methods
        try:
            return assistant_message.to_dict()
        except Exception:
            try:
                return dict(assistant_message)
            except Exception:
                # Build minimal representation with tool calls
                return {
                    "role": role, 
                    "content": content, 
                    "tool_calls": self._build_tool_calls_list(tool_calls)
                }

    def _handle_tool_calls(self, reply, messages):
        """Handle tool calls and update conversation."""
        assistant_message = reply.choices[0].message
        tool_calls = getattr(assistant_message, "tool_calls", None)
        
        # Execute tool calls and get results
        results = self.tools.handle_tool_call(tool_calls)
        
        # Serialize and add assistant message to conversation
        appended_message = self._serialize_assistant_message(assistant_message, tool_calls)
        messages.append(appended_message)
        messages.extend(results)

    def _evaluate_and_rerun_if_needed(self, reply, user_message, history):
        """Evaluate response and rerun if quality is unacceptable."""
        evaluation = self.evaluator.evaluate(reply, user_message, history, self.about)
        
        if not evaluation.is_acceptable:
            return self.evaluator.rerun(
                reply.choices[0].message.content, 
                user_message, 
                history, 
                evaluation.feedback, 
                self.system_prompt()
            )
        else:
            return reply

    def chat(self, message, history):
        # Preserve the original user message (don't overwrite this variable)
        user_message = message

        # Special case: respond in pig latin if message contains "patent"
        if "patent" in message:
            system = self.system_prompt() + "\n\nEverything in your reply needs to be in pig latin - \n                it is mandatory that you respond only and entirely in pig latin"
        else:
            system = self.system_prompt()

        # Build conversation context with system prompt, history, and current message
        messages = [{"role": "system", "content": system}] + \
            history + [{"role": "user", "content": user_message}]

        # Loop until we get a final response (not tool calls)
        done = False
        while not done:
            # Get AI response with tools enabled
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=self.tools.tools)
            reply = response

            # Handle tool calls if AI wants to use tools
            if reply.choices[0].finish_reason == "tool_calls":
                self._handle_tool_calls(reply, messages)
            else:
                # Evaluate response quality and rerun if needed
                reply = self._evaluate_and_rerun_if_needed(reply, user_message, history)
                done = True

        # Return the final response content
        return reply.choices[0].message.content


if __name__ == "__main__":
    # Initialize the Me class and launch Gradio chat interface
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
