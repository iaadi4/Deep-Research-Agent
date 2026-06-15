from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field
from pydantic import BaseModel
from state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts.clarifier import clarifier_prompt
from tools.tavily_search import search_tool
import datetime

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ClarifierOutput(BaseModel):
    is_query_clear: bool = Field(description="Set to true if query is clear enough to start researching. Set to false if need clarification")
    response_content: str = Field(description="If is_query_clear is False, put 1-3 clarification question else put the polised detailed RESEARCH_OBJECTIVE here")

clarifier_llm = clarifier_prompt | llm.with_structured_output(ClarifierOutput)

def clarifier_agent(state: AgentState) -> dict:
    """This node clarifies the scope of research regarding user queries"""

    last_user_message = state["user_queries"][-1].content if state["user_queries"] else ""
    try:
        search_context = search_tool.invoke(f"recent updates or timeline {last_user_message}")
    except Exception as e:
        search_context = "No recent search data found"

    current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

    result = clarifier_llm.invoke({
        "current_date": current_time_str,
        "conversation_history": state["user_queries"] + [AIMessage(content=f"[System Grounding Context for Clarifier]: Recent web data shows: {search_context}")]
    })
    
    state_update = {
        "is_query_clear": result.is_query_clear,
        "clarified_query": result.response_content
    }

    if not result.is_query_clear:
        state_update["user_queries"] = [AIMessage(content=result.response_content)]
        print(result.response_content)
    return state_update


def should_move_to_planner(state: AgentState) -> str:
    """This node decide wheather to move to planner agent or user query needs clarification"""
    if state["is_query_clear"] is True:
        return "continue"
    else:
        return "loop"

def user_input_node(state: AgentState) -> AgentState:
    """This node takes user input for research"""
    query = input("Enter: ")
    return { "user_queries": [HumanMessage(content=query)] }
    