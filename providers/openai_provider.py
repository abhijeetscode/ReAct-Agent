from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from .provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    async def agenerate(
        self, messages: list[dict[str, str]], **kwargs
    ) -> ChatCompletion:
        response = self.client.chat.completions.create(
            messages=messages,  # type: ignore
            model=self.model,
        )
        return response
