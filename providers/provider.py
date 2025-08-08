from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> str:
        pass
