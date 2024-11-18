import os

from openai import OpenAI
import requests


class GPT:
    def __init__(self, api_key, max_tokens, model='gpt-4o-mini', system_prompt='You are a helpful assistant.'):
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.queue = False
        self.model = model
        self.system_prompt = {
            'role': 'system',
            'content': system_prompt
        }
        self.conversation = [self.system_prompt]
        self.cost = 0

    def start_queue(self):
        self.queue = True

    def end_queue(self):
        self.queue = False
        self.conversation.clear()
        self.conversation.append(self.system_prompt)

    def _construct_message_(self, text):
        current_message = {
            'role': 'user',
            'content': text
        }
        # 如果是上下文模式 就要添加上下文内容
        if self.queue:
            copy = self.conversation.copy()
            copy.append(current_message)
            return copy
        return [self.system_prompt, current_message]

    # 有上下文 但是不保存当前对话
    def ask_without_save(self, text):
        message = self._construct_message_(text)
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        self._calculate_cost_(response.usage.model_dump())
        return content

    def ask(self, text):
        content = self.ask_without_save(text)
        # 保存对话
        self.add_user_context(text)
        self.add_assistant_context(content)
        return content

    def ask_single(self, text):
        temp = self.queue
        self.queue = False
        content = self.ask(text)
        self.queue = temp
        return content

    def add_user_context(self, text):
        if self.queue:
            self.conversation.append({
                'role': 'user',
                'content': text
            })

    def add_assistant_context(self, text):
        if self.queue:
            self.conversation.append({
                'role': 'assistant',
                'content': text
            })

    def _calculate_cost_(self, usage):
        self.cost += usage['completion_tokens'] * 3 + usage['prompt_tokens']

    def get_cost_tokens(self):
        return self.cost
