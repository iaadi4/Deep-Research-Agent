from typing import Literal
from pydantic import BaseModel, Field
import operator
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class ResearchItem(BaseModel):
    source: str = Field(description="source of the information gathered by agent")
    title: str = Field(description="title of the information")
    content: str = Field(description="raw content/body of the information")
    confidence: Literal["high", "medium", "low"] = Field(
        description="confidence in this finding, based on source reliability and how directly it answers the guiding question"
    )

class ResearchRecord(ResearchItem):
    agent_id: str = Field(description="ID of the task/agent that produced this item")
    task_title: str = Field(description="Title of the task that produced this item")

class ResearchTasks(BaseModel):
    agent_id: str = Field(description="Short snake_case identifier for this task, e.g. 'competitor_pricing' or 'regulatory_landscape_eu'. Used to tag results during synthesis.")
    title: str = Field(description="Short human-readable title for this research task")
    instructions: str = Field(description="A complete, self-contained research brief for an isolated sub-agent, including all necessary context, scope, and what's out of bounds")
    guiding_questions: list[str] = Field(description="2-4 specific sub-questions this agent should answer through research")

class AgentState(TypedDict):
    # user queries (including clarifier agent answers)
    user_queries: Annotated[Sequence[BaseMessage], add_messages]
    is_query_clear: bool

    # clarified query from Clarifier agent 
    clarified_query: str

    # Research plan from Planner agent
    research_brief: str
    research_tasks: list[dict]

    # Data from multiple Research agent
    task: ResearchTasks
    raw_research_data: Annotated[list[ResearchItem], operator.add]

    # Structured compiled data from sub Research agent by Synthesizer agent
    compiled_research: str

    # Judge agent output
    missing_items: list[str]
    judge_comments: str
    is_satisfied: bool
    iteration_count: int

    # Report agent final report
    final_report: str
