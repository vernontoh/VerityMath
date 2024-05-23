import re
import time
import openai
from utils import GenericRuntime


class ProgramChatInterface:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        system_message="You are a helpful assistant",
        openai_key=None,
    ) -> None:

        self.model = model
        self.system_message = system_message
        self.runtime = GenericRuntime()
        openai.api_key = openai_key

    def call_chat_gpt(self, messages, model, temperature, top_p, max_tokens):
        wait = 1
        while True:
            try:
                ans = openai.ChatCompletion.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=1,
                )
                return ans.choices[0]["message"]["content"]
            except openai.error.RateLimitError as e:
                time.sleep(min(wait, 60))
                wait *= 2

    def generate(self, prompt, temperature=0.0, top_p=1.0, max_tokens=512):
        message = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        program = self.call_chat_gpt(
            messages=message,
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return program

    def run(self, prompt, temperature=0.0, top_p=1.0, max_tokens=512):
        program = self.generate(
            prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        code, ans, error = self.runtime.run_code(
            code_gen=program, answer_expr="solution()"
        )
        return program, code, ans


class ClassificationChatInterface:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        system_message="You are a helpful assistant",
        openai_key=None,
    ) -> None:

        self.model = model
        self.system_message = system_message
        self.runtime = GenericRuntime()
        openai.api_key = openai_key

    def call_chat_gpt(self, messages, model, temperature, top_p, max_tokens):
        wait = 1
        while True:
            try:
                ans = openai.ChatCompletion.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=1,
                )
                return ans.choices[0]["message"]["content"]
            except Exception as e:
                print(e)
                time.sleep(min(wait, 60))
                wait *= 2

    def generate(self, prompt, temperature=0.0, top_p=1.0, max_tokens=512):
        message = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        explanation = self.call_chat_gpt(
            messages=message,
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return explanation

    def run(self, prompt, temperature=0.0, top_p=1.0, max_tokens=512):
        explanation = self.generate(
            prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        label = re.sub("\W+", "", explanation.split("Answer:")[-1]).lower()
        assert label in ["yes", "no"]

        return explanation, label
