from ollama import chat, ChatResponse
from .provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        self.model = model

    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> str:
        response: ChatResponse = chat(model=self.model, messages=messages)
        return response.message.content or ""
