from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from tools.firecrawl_scrapping import scrape_tool
from tools.tavily_search import search_tool
from tools.summarizer import summarize_tool
from typing import TypedDict, Annotated, Sequence

tools = [scrape_tool, search_tool, summarize_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def first_node(state: AgentState) -> AgentState:
    """This node answers user query using appropriate tools"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """This node decide wheather to use tools or end"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("first_node", first_node)
graph.add_node("research_tools", tool_node)

graph.set_entry_point("first_node")

graph.add_conditional_edges(
    "first_node",
    should_continue,
    {
        "continue": "research_tools",
        "end": END
    }
)

graph.add_edge("research_tools", "first_node")

agent = graph.compile()
result = agent.invoke({"messages": [("user", "what is price of gold today?")]})
print(result["messages"][-1].content)
