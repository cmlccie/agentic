# LangGraph + FastMCP Integration

`fastmcp.py` provides `mcp_tools()` — a helper that converts a FastMCP `Client`'s tool list into LangChain `StructuredTool`s for use with LangGraph agents.

## Usage Patterns

### 1. Simple — one server, one agent invocation

```python
import asyncio
from fastmcp import Client
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from agentic.langchain.fastmcp import mcp_tools

MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MCP_URL = "http://localhost:8000/mcp"

async def simple_agent(question: str) -> str:
    async with Client(MCP_URL) as client:
        agent = create_agent(MODEL, await mcp_tools(client))
        result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
        return result["messages"][-1].content

if __name__ == "__main__":
    print(asyncio.run(simple_agent("What tools do you have available?")))
```

### 2. Resources — inject resource context into the prompt

```python
import asyncio
from fastmcp import Client
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from agentic.langchain.fastmcp import mcp_tools

MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MCP_URL = "http://localhost:8000/mcp"

async def agent_with_resource_context(question: str) -> str:
    async with Client(MCP_URL) as client:
        config = await client.read_resource("file:///config/app.json")
        context = config[0].text if config else ""

        agent = create_agent(MODEL, await mcp_tools(client))
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        return result["messages"][-1].content

if __name__ == "__main__":
    print(asyncio.run(agent_with_resource_context("Summarize the app config.")))
```

### 3. Multiple servers — merge tools from concurrent clients

```python
import asyncio
from fastmcp import Client
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from agentic.langchain.fastmcp import mcp_tools

MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def multi_server_agent(question: str) -> str:
    async with (
        Client("http://localhost:8001/mcp") as files,
        Client("http://localhost:8002/mcp") as db,
    ):
        tools = [
            *await mcp_tools(files, server_prefix="files"),
            *await mcp_tools(db, server_prefix="db"),
        ]
        agent = create_agent(MODEL, tools)
        result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
        return result["messages"][-1].content

if __name__ == "__main__":
    print(asyncio.run(multi_server_agent("List files and show recent DB records.")))
```

### 4. LangGraph `StateGraph` — custom graph with MCP tools

```python
import asyncio
from typing import Annotated, TypedDict
from fastmcp import Client
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from agentic.langchain.fastmcp import mcp_tools

MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MCP_URL = "http://localhost:8000/mcp"

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def custom_graph_agent(question: str) -> str:
    async with Client(MCP_URL) as client:
        tools = await mcp_tools(client)
        llm_with_tools = MODEL.bind_tools(tools)

        async def agent_node(state: State) -> dict:
            return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

        graph = StateGraph(State)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        result = await graph.compile().ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        return result["messages"][-1].content

if __name__ == "__main__":
    print(asyncio.run(custom_graph_agent("What tools do you have available?")))
```
