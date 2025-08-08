import asyncio
from dotenv import load_dotenv

from providers.ollama_providers import OllamaProvider
from agent import Agent
from tools import get_weather, calculate_compound_interest

load_dotenv()


async def main():
    agent = Agent(llm=OllamaProvider(model="gemma3:27b"))
    await agent.arun(
        query="If I invest $5000 at 3.5% compounded quarterly for 15 years, how much will I have? and What's the weather like in London?",
        tools=[get_weather, calculate_compound_interest],
    )


if __name__ == "__main__":
    asyncio.run(main())
