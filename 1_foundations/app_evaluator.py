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

load_dotenv(override=True)


class Me:
    class Evaluation(BaseModel):
        is_acceptable: bool
        feedback: str

    def __init__(self):
        self.openai = OpenAI()
        self.gemini = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.name = "Don Kukral"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

        self.about = {'name': self.name,
                      'summary': self.summary, 'linkedin': self.linkedin}
        self.evaluator = Evaluator()
        self.tools = Tools()

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def chat(self, message, history):
        # preserve the original user message (don't overwrite this variable)
        user_message = message

        if "patent" in message:
            system = self.system_prompt() + "\n\nEverything in your reply needs to be in pig latin - \n                it is mandatory that you respond only and entirely in pig latin"
        else:
            system = self.system_prompt()

        messages = [{"role": "system", "content": system}] + \
            history + [{"role": "user", "content": user_message}]

        done = False
        while not done:

            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=self.tools.tools)
            reply = response

            if reply.choices[0].finish_reason == "tool_calls":
                assistant_message = reply.choices[0].message

                tool_calls = getattr(assistant_message, "tool_calls", None)

                results = self.tools.handle_tool_call(tool_calls)

                # Normalize the assistant message so content is a JSON-serializable string
                role = getattr(assistant_message, "role", "assistant")
                content = getattr(assistant_message, "content", None)
                if not isinstance(content, str):
                    try:
                        content = json.dumps(content)
                    except Exception:
                        content = str(content)

                # If the assistant message included tool_calls, we must preserve that
                # metadata when appending the assistant message to the conversation.
                # The OpenAI API requires a preceding assistant message with the
                # `tool_calls` field in order to accept subsequent messages with
                # role "tool".
                appended_message = None
                if tool_calls:
                    # Try to convert the assistant_message to a plain dict using
                    # available helpers on the object. Fall back to constructing
                    # a minimal dict with tool_calls info.
                    try:
                        # Many OpenAI response objects have a `to_dict()` method.
                        appended_message = assistant_message.to_dict()
                    except Exception:
                        try:
                            appended_message = dict(assistant_message)
                        except Exception:
                            # Build a minimal serializable representation
                            tc_list = []
                            for tc in tool_calls:
                                try:
                                    tc_dict = {
                                        "id": getattr(tc, "id", None),
                                        "function": {
                                            "name": getattr(getattr(tc, "function", None), "name", None),
                                            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                                        },
                                    }
                                except Exception:
                                    tc_dict = {"id": None}
                                tc_list.append(tc_dict)

                            appended_message = {
                                "role": role, "content": content, "tool_calls": tc_list}
                else:
                    appended_message = {"role": role, "content": content}

                messages.append(appended_message)
                messages.extend(results)
            else:
                evaluation = self.evaluator.evaluate(
                    reply, user_message, history, self.about)
                print("Evaluation:", evaluation.is_acceptable,
                      "\nFeedback:", evaluation.feedback)
                if not evaluation.is_acceptable:
                    print("rejecting response due to quality control",
                          evaluation.feedback)
                    print("About:", self.about)
                    # pass the original user_message (string) into rerun
                    reply = self.evaluator.rerun(
                        reply.choices[0].message.content, user_message, history, evaluation.feedback, self.system_prompt())
                else:
                    print("accepting response")

                print("Reply:", reply.choices[0].message.content,
                      "\nFinish reason:", reply.choices[0].finish_reason)
                done = True

        return reply.choices[0].message.content


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
