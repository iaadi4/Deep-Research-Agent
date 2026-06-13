from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field
from pydantic import BaseModel
from state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts.clarifier import clarifier_prompt

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ClarifierOutput(BaseModel):
    is_query_clear: bool = Field(description="Set to true if query is clear enough to start researching. Set to false if need clarification")
    response_content: str = Field(description="If is_query_clear is False, put 1-3 clarification question else put the polised detailed RESEARCH_OBJECTIVE here")

clarifier_llm = clarifier_prompt | llm.with_structured_output(ClarifierOutput)

def clarifier_agent(state: AgentState) -> dict:
    """This node clarifies the scope of research regarding user queries"""
    result = clarifier_llm.invoke({
        "conversation_history": state["user_queries"]
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
    