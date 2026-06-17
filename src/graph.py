from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from state import AgentState 
from agents.clarifier_agent import clarifier_agent, user_input_node, should_move_to_planner
from agents.planner_agent import planner_agent
from agents.researcher_agent import researcher_agent
from langgraph.types import Send

def dispatch_research_agents(state: AgentState):
    return [
        Send("researcher_agent", {"task": task})
        for task in state["research_tasks"]
    ]

graph = StateGraph(AgentState)
graph.add_node("user_input_node", user_input_node)
graph.add_node("clarifier_agent", clarifier_agent)
graph.add_node("planner_agent", planner_agent)
graph.add_node("researcher_agent", researcher_agent)

graph.set_entry_point("user_input_node")
graph.add_edge("user_input_node", "clarifier_agent")
graph.add_conditional_edges("planner_agent", dispatch_research_agents, ["researcher_agent"])
graph.set_finish_point("researcher_agent")

graph.add_conditional_edges(
    "clarifier_agent",
    should_move_to_planner,
    {
        "continue": "planner_agent",
        "loop": "user_input_node"
    }
)

deep_research_agent = graph.compile()
print(deep_research_agent.get_graph().draw_mermaid())

deep_research_agent.invoke({})
