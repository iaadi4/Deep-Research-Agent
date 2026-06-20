from pydantic import Field, BaseModel
from llm import get_llm
from state import AgentState, ResearchTasks
from prompts.planner_prompt import planner_prompt
import datetime

class PlannerOutput(BaseModel):
    research_brief: str = Field(description="2-4 sentence synthesis of the overall objective and how the tasks below relate, for the final synthesis agent")
    research_tasks: list[ResearchTasks] = Field(description="2-6 independent research tasks to run in parallel")

llm = get_llm(temperature=0.5)
planner_llm = planner_prompt | llm.with_structured_output(PlannerOutput, method="json_mode")

def planner_agent(state: AgentState) -> dict:
    """This agent takes input from clarifier agent and decomposes the prompt into subtasks."""

    current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
    result = planner_llm.invoke({
        "current_date": current_date_str,
        "clarified_query": state["clarified_query"],
    })

    print(result)
    
    return {
        "research_brief": result.research_brief,
        "research_tasks": result.research_tasks,  
    }