from dotenv import load_dotenv
from openai import OpenAI
# import json
import os
# import requests
from pydantic import BaseModel


class Evaluator:
    def __init__(self):

        load_dotenv(override=True)

        self.openai = OpenAI()
        self.gemini = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        pass

    class Evaluation(BaseModel):
        is_acceptable: int
        feedback: str

    def evaluator_system_prompt(self, about):
        evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {about["name"]} and is representing {about["name"]} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {about["name"]} in the form of their summary and LinkedIn details. Here's the information:"

        evaluator_system_prompt += f"\n\n## Summary:\n{about["summary"]}\n\n## LinkedIn Profile:\n{about["linkedin"]}\n\n"
        evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
        return evaluator_system_prompt

    def evaluator_user_prompt(self, reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    def evaluate(self, reply, message, history, about) -> Evaluation:
        messages = [{"role": "system", "content": self.evaluator_system_prompt(
            about)}] + [{"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}]
        response = self.gemini.beta.chat.completions.parse(
            model="gemini-2.0-flash", messages=messages, response_format=self.Evaluation)
        return response.choices[0].message.parsed

    def rerun(self, reply, message, history, feedback, system_prompt):
        updated_system_prompt = system_prompt + \
            "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_system_prompt}
                    ] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages)
        # return response.choices[0].message.content
        return response
