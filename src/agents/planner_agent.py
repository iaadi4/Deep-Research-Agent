from pydantic import Field, BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState, ResearchTasks
from prompts.planner_prompt import planner_prompt
import datetime

class PlannerOutput(BaseModel):
    research_brief: str = Field(description="2-4 sentence synthesis of the overall objective and how the tasks below relate, for the final synthesis agent")
    research_tasks: list[ResearchTasks] = Field(description="2-6 independent research tasks to run in parallel")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
planner_llm = planner_prompt | llm.with_structured_output(PlannerOutput)

def planner_agent(state: AgentState) -> dict:
    """This agent takes input from clarifier agent and decomposes the prompt into subtasks."""

    current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
    result = planner_llm.invoke({
        "current_date": current_date_str,
        "clarified_query": state["clarified_query"],
    })
    
    return {
        "research_brief": result.research_brief,
        "research_tasks": result.research_tasks,  
    }