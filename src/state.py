import operator
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class ResearchItem(TypedDict):
    source: str
    title: str
    content: str
    confidence: str

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
    raw_research_data: Annotated[list[ResearchItem], operator.add]

    # Judge agent output
    missing_items: list[str]
    judge_comments: str
    is_satisfied: bool
    iteration_count: int

    # Report agent final report
    final_report: str
