import inspect
import json
import logging
import os
from typing import Any, Callable
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.utils.function_calling import convert_to_openai_function
from providers.ollama_providers import LLMProvider
from system_prompt import SYSTEM_PROMPT
from utils import load_json_string

# Configure LangSmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "react-agent")
# Set LANGCHAIN_API_KEY in your environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: list[dict[str, str]]
    observation: str
    tools_output: list[Any]


class ThinkOutput(TypedDict):
    tool_name: str
    args: dict[str, Any]


class GraphBuilder:
    def __init__(self, llm: LLMProvider, tools_map: dict[str, Callable]) -> None:
        self.llm = llm
        self.tools_map: dict[str, Callable] = tools_map
        self.graph_builder = StateGraph(State)

    async def _init(self, state: State) -> State:
        logger.info("Initializing agent state")
        logger.debug(f"Initial state: {state}")
        return state

    async def _think(self, state: State) -> State:
        logger.info("Agent thinking phase started")
        logger.debug(f"Input messages count: {len(state['messages'])}")

        response: str = await self.llm.agenerate(messages=state["messages"])
        logger.info(f"LLM response generated: {response[:100]}...")

        state["messages"].append({"role": "assistant", "content": response})
        logger.debug(f"Updated messages count: {len(state['messages'])}")
        return state

    async def _act(self, state: State) -> State:
        logger.info("Agent action phase started")
        last_message: str = state["messages"][-1]["content"]
        logger.debug(f"Processing message: {last_message[:100]}...")

        try:
            output: ThinkOutput = load_json_string(last_message)  # type: ignore
            if not output:
                raise ValueError("Empty or invalid JSON response")
            logger.info(
                f"Parsed tool call: {output['tool_name']} with args: {output['args']}"
            )
        except ValueError as e:
            logger.error(f"JSON parsing failed: {e}")
            state["observation"] = "JSON string is empty. Please provide valid JSON"
        except Exception as e:
            error_message = f"Invalid JSON {str(e)}"
            logger.error(f"JSON error: {error_message}")
            state["observation"] = error_message

        else:
            name = output["tool_name"]
            args: str | dict[str, Any] = output["args"]
            if args and isinstance(args, str):
                args = load_json_string(args)

            tool: Callable | None = self.tools_map.get(name)
            if not tool:
                logger.error(f"Tool not found: {name}")
                state["observation"] = f"Error: Invalid Tool {name}"
            else:
                logger.info(f"Executing tool: {name}")
                if inspect.iscoroutinefunction(tool):
                    try:
                        result = await tool(**args)  # type: ignore
                        logger.info(f"Async tool {name} executed successfully")
                    except Exception as e:
                        error_message = f"Error to execute tool {name} {str(e)}"
                        logger.error(f"Async tool execution failed: {error_message}")
                        state["observation"] = error_message

                    else:
                        state["messages"].append(
                            {
                                "role": "tool",
                                "name": name,
                                "content": f"Tool result: {result}",
                            }
                        )
                        state["observation"] = f"{name} executed successfully"
                        state["tools_output"].append(result)
                        logger.debug(f"Tool result: {str(result)[:200]}...")
                else:
                    try:
                        result = tool(**args)
                        logger.info(f"Sync tool {name} executed successfully")
                    except Exception as e:
                        error_message = f"Error to execute tool {name} {str(e)}"
                        logger.error(f"Sync tool execution failed: {error_message}")
                        state["observation"] = f"Error to execute tool {name} {str(e)}"
                    else:
                        state["messages"].append(
                            {
                                "role": "tool",
                                "name": name,
                                "content": f"Tool result: {str(result)}",
                            }
                        )
                        state["observation"] = f"{name} executed successfully"
                        state["tools_output"].append(result)
                        logger.debug(f"Tool result: {str(result)[:200]}...")

        logger.info(f"Action phase completed. Observation: {state['observation']}")
        return state

    async def _generate_final_answer(self, state: State) -> str:
        logger.info("Generating final answer")
        tools_result = state["tools_output"]
        user_query: str = state["messages"][1]["content"]
        logger.debug(f"User query: {user_query}")
        logger.debug(f"Tools output count: {len(tools_result)}")

        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": f"""
                                          Generate a response for User query: {user_query}  by looking at Output:: {tools_result}
                                            Response should be polite and you are giving response to human and to the point""",
            }
        ]
        response = await self.llm.agenerate(messages=messages)
        logger.info(f"Final answer generated: {response[:100]}...")
        return response

    async def _should_end(self, state: State) -> bool:
        logger.info("Checking if agent should end")
        last_message = state["messages"][-1]["content"]
        logger.debug(f"Last message: {last_message[:100]}...")

        try:
            output: ThinkOutput = load_json_string(last_message)  # type: ignore
            if not output:
                raise ValueError("Empty or invalid JSON response")
            logger.debug(f"Parsed output: {output}")
        except ValueError as e:
            logger.warning(f"JSON parsing failed in _should_end: {e}")
            state["observation"] = "JSON string is empty. Please provide valid JSON"
        except Exception as e:
            error_message = f"Invalid JSON {str(e)}"
            logger.warning(f"JSON error in _should_end: {error_message}")
            state["observation"] = error_message
        else:
            if output and "tool_name" in output and output["tool_name"] == "END":
                logger.info("END tool detected, generating final answer")
                final_answer: str = await self._generate_final_answer(state=state)
                xx = {"role": "assistant", "content": final_answer.strip()}
                state["messages"].append(xx)
                logger.info("Agent execution completed")
                return True

        logger.info("Agent will continue to next iteration")
        return False

    def _create_graph(self):
        self.graph_builder.add_node("init", self._init)
        self.graph_builder.add_node("think", self._think)
        self.graph_builder.add_node("act", self._act)

        self.graph_builder.add_edge(START, "init")
        self.graph_builder.add_edge("init", "think")
        self.graph_builder.add_edge("think", "act")
        self.graph_builder.add_conditional_edges(
            "act", self._should_end, {True: END, False: "think"}
        )
        return self.graph_builder.compile()

    def __call__(self) -> Any:
        return self._create_graph()


class Agent:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.tools_map: dict[str, Callable] = dict()
        self.graph_builder = GraphBuilder(self.llm, self.tools_map)

    def _get_string_repr_of_tools(self, tools: list[Callable]) -> str:
        descs = [json.dumps(convert_to_openai_function(tool)) for tool in tools]
        return "\n".join(descs)

    def _setup(self, tools: list[Callable]):
        self.tools_map = {tool.__name__: tool for tool in tools}
        self.graph_builder.tools_map = self.tools_map
        self.graph = self.graph_builder._create_graph()

    async def arun(self, query: str, tools: list[Callable]) -> None:
        logger.info(f"Starting agent run with query: {query}")
        logger.info(f"Available tools: {[tool.__name__ for tool in tools]}")

        self._setup(tools=tools)
        tools_repr = self._get_string_repr_of_tools(tools=tools)
        system_prompt = SYSTEM_PROMPT.format(available_tools=tools_repr, task=query)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": query})

        logger.debug(f"System prompt: {system_prompt[:200]}...")

        async for event in self.graph.astream(
            {"messages": messages, "observation": "", "tools_output": []}
        ):  # type: ignore
            for node_name, value in event.items():
                logger.info(f"Node '{node_name}' executed")
                print(value["messages"][-1]["content"])

        logger.info("Agent run completed")
