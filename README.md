# ReAct Agent

A Python implementation of the ReAct (Reasoning and Acting) pattern for building AI agents that can think and act using external tools.

## Overview

This agent uses the ReAct framework to:
- **Think**: Reason about the problem and decide what action to take
- **Act**: Execute tools to gather information or perform tasks
- **Observe**: Process the results and continue reasoning

The agent is built using LangGraph for state management and supports multiple LLM providers.

## Features

- 🧠 ReAct pattern implementation with thinking and acting cycles
- 🔧 Tool integration system for external function calls
- 🌐 Multiple LLM provider support (Ollama, OpenAI)
- 📊 State-based conversation flow with LangGraph
- ⚡ Async/await support for concurrent operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abhijeetscode/ReAct-Agent.git
cd ReAct-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Basic Example

```python
import asyncio
from providers.ollama_providers import OllamaProvider
from agent import Agent
from tools import get_weather, calculate_compound_interest

async def main():
    agent = Agent(llm=OllamaProvider(model="gemma3:27b"))
    await agent.arun(
        query="If I invest $5000 at 3.5% compounded quarterly for 15 years, how much will I have? and What's the weather like in London?",
        tools=[get_weather, calculate_compound_interest],
    )

asyncio.run(main())
```

### Available Tools

- `get_weather`: Get current weather information for any city
- `calculate_compound_interest`: Calculate compound interest for investments

## Project Structure

```
.
├── agent.py              # Main Agent class and ReAct implementation
├── main.py               # Example usage
├── providers/            # LLM provider implementations
│   ├── ollama_providers.py
│   ├── openai_provider.py
│   └── provider.py
├── tools.py              # Tool definitions
├── system_prompt.py      # System prompt templates
└── utils.py              # Utility functions
```

## How It Works

1. **Initialization**: Agent is created with an LLM provider and tools
2. **System Prompt**: Agent receives instructions on how to use the ReAct pattern
3. **Think-Act Loop**: 
   - Agent thinks about the problem and outputs JSON with tool name and arguments
   - Agent acts by executing the specified tool
   - Agent observes the results and continues thinking
4. **Final Answer**: When complete, agent generates a final response

## LLM Providers

### Ollama
```python
from providers.ollama_providers import OllamaProvider
agent = Agent(llm=OllamaProvider(model="gemma3:27b"))
```

### OpenAI
```python
from providers.openai_provider import OpenAIProvider
agent = Agent(llm=OpenAIProvider(model="gpt-4"))
```


## License

MIT License