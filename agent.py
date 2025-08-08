import inspect
import json
import asyncio
from typing import Any, Callable
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.utils.function_calling import convert_to_openai_function
from providers.ollama_providers import LLMProvider
from system_prompt import SYSTEM_PROMPT
from utils import load_json_string

class State(TypedDict):
    messages: list[dict[str, str]]
    observation: str
    tools_output:list[Any]

class ThinkOutput(TypedDict):
    tool_name: str
    args: dict[str, Any]







class GraphBuilder:
    def __init__(self, llm: LLMProvider, tools_map: dict[str, Callable]) -> None:
        self.llm = llm
        self.tools_map: dict[str, Callable] = tools_map
        self.graph_builder = StateGraph(State)
        
    async def _init(self, state: State) -> State:
        return state
    
    async def _think(self, state: State) -> State:
        response: str = await self.llm.agenerate(messages=state["messages"])
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    async def _act(self, state: State) -> State:
        last_message: str = state["messages"][-1]["content"]
        try:
            output:ThinkOutput = load_json_string(last_message) # type: ignore
            if not output:
                raise ValueError(f"Empty or invalid JSON response")
        except ValueError as e:
            state["observation"] = "JSON string is empty. Please provide valid JSON"
        except Exception as e:
            error_message = f"Invalid JSON {str(e)}"
            state["observation"] = error_message

        else:
            name = output["tool_name"]
            args:str | dict[str, Any] = output["args"]
            if args and isinstance(args, str):
                args = load_json_string(args)
            tool: Callable | None = self.tools_map.get(name)
            if not tool:
                state["observation"] = f"Error: Invalid Tool {name}"
            else:

                if inspect.iscoroutinefunction(tool):
                    try:
                        result = await tool(**args) #type: ignore
                    except Exception as e:
                        error_message = f"Error to execute tool {name} {str(e)}"
                        state["observation"] = error_message
                     
                    else:
                        state["messages"].append({
                            "role": "tool",
                            "name": name,
                            "content": f"Tool result: {result}"
                        })
                        state["observation"] = f"{name} executed successfully"
                        state["tools_output"].append(result)
                else:
                    try:
                        result = tool(**args)
                    except Exception as e:
                        error_message = f"Error to execute tool {name} {str(e)}"
                        state["observation"] = f"Error to execute tool {name} {str(e)}"
                    else:
                        state["messages"].append({
                            "role": "tool",
                            "name": name,
                            "content": f"Tool result: {str(result)}"
                        })
                        state["observation"] = f"{name} executed successfully"
                        state["tools_output"].append(result)
        return state
    async def _generate_final_answer(self, state: State)->str:
        tools_result = state["tools_output"]
        user_query: str = state["messages"][1]["content"]
        messages:list[dict[str, str]] = [{"role": "user", "content": f"""
                                          Generate a response for User query: {user_query}  by looking at Output:: {tools_result}
                                            Response should be polite and you are giving response to human and to the point"""}]
        response = await self.llm.agenerate(messages=messages)
        return response
    async def _should_end(self, state: State) -> bool:
        last_message = state["messages"][-1]["content"]
        try:
            output: ThinkOutput = load_json_string(last_message) # type: ignore
            if not output:
                raise ValueError("Empty or invalid JSON response")
        except ValueError as e:
            state["observation"] = "JSON string is empty. Please provide valid JSON"
        except Exception as e:
            error_message = f"Invalid JSON {str(e)}"
            state["observation"] = error_message
        else:
            if output and "tool_name" in output and  output["tool_name"] == "END":
                final_answer: str = await self._generate_final_answer(state=state)
                xx = {"role": "assistant", "content": final_answer.strip()}
                state["messages"].append(xx)
                return True
        return False
    
    def _create_graph(self):
        self.graph_builder.add_node("init", self._init)
        self.graph_builder.add_node("think", self._think)
        self.graph_builder.add_node("act", self._act)

        self.graph_builder.add_edge(START, "init")
        self.graph_builder.add_edge("init", "think")
        self.graph_builder.add_edge("think", "act")
        self.graph_builder.add_conditional_edges("act", self._should_end, {True: END, False: "think"})        
        return self.graph_builder.compile()

    def __call__(self) -> Any:
        return self._create_graph()


class Agent:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.tools_map: dict[str, Callable] = dict()
        self.graph_builder = GraphBuilder(self.llm, self.tools_map)

    
    def _get_string_repr_of_tools(self, tools: list[Callable])->str:
        descs = [json.dumps(convert_to_openai_function(tool)) for tool in tools]
        return "\n".join(descs)
    
    def _setup(self, tools: list[Callable]):
        self.tools_map = {tool.__name__: tool for tool in tools}
        self.graph_builder.tools_map = self.tools_map
        self.graph = self.graph_builder._create_graph()

    async def arun(self, query: str, tools: list[Callable])->None:
        self._setup(tools=tools)
        tools_repr = self._get_string_repr_of_tools(tools=tools)
        system_prompt = SYSTEM_PROMPT.format(available_tools=tools_repr, task=query)
        messages:list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": query})
        async for event in self.graph.astream({
            "messages": messages,
            "observation": "",
            "tools_output": []
        }): # type: ignore
            for node_name, value in event.items():
                print(value["messages"][-1]["content"])
                
            