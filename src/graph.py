from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from state import AgentState 
from agents.clarifier.agents import clarifier_agent, user_input_node, should_move_to_planner
from agents.planner.agents import planner_agent

graph = StateGraph(AgentState)
graph.add_node("user_input_node", user_input_node)
graph.add_node("clarifier_agent", clarifier_agent)
graph.add_node("planner_agent", planner_agent)

graph.set_entry_point("user_input_node")
graph.add_edge("user_input_node", "clarifier_agent")
graph.set_finish_point("planner_agent")

graph.add_conditional_edges(
    "clarifier_agent",
    should_move_to_planner,
    {
        "continue": "planner_agent",
        "loop": "user_input_node"
    }
)

research_agent = graph.compile()

research_agent.invoke({"messages": []})
