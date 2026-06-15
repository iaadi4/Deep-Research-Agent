from pydantic import Field
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
from prompts.planner import planner_prompt
import datetime

class ResearchTasks(BaseModel):
    agent_id: str = Field(description="Short snake_case identifier for this task, e.g. 'competitor_pricing' or 'regulatory_landscape_eu'. Used to tag results during synthesis.")
    title: str = Field(description="Short human-readable title for this research task")
    instructions: str = Field(description="A complete, self-contained research brief for an isolated sub-agent, including all necessary context, scope, and what's out of bounds")
    guiding_questions: list[str] = Field(description="2-4 specific sub-questions this agent should answer through research")

class PlannerOutput(BaseModel):
    research_brief: str = Field(description="2-4 sentence synthesis of the overall objective and how the tasks below relate, for the final synthesis agent")
    research_tasks: list[ResearchTasks] = Field(description="2-6 independent research tasks to run in parallel")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

planner_llm = planner_prompt | llm.with_structured_output(PlannerOutput)

def planner_agent(state: AgentState) -> dict:
    """This agent takes input from clarifier agent and decomposes the prompt in subtasks and
    source needed to get information
    """

    current_date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

    result = planner_llm.invoke({
        "current_date": current_date_str,
        "clarified_query": state["clarified_query"]
    })

    return {
        "research_brief": result.research_brief,
        "research_tasks": [task.model_dump() for task in result.research_tasks]
    }