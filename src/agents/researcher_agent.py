from pydantic import BaseModel, Field
from llm import get_llm
from langgraph.prebuilt import create_react_agent
from state import AgentState, ResearchItem, ResearchRecord
from prompts.researcher_prompt import researcher_prompt
from tools.firecrawl_scrapping_tool import scrape_tool
from tools.summarizer_tool import summarize_tool
from tools.tavily_search_tool import search_tool
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

class ResearchAgentOutput(BaseModel):
    items: list[ResearchItem] = Field(description="Discrete pieces of information gathered, one per source/claim")

llm = get_llm(temperature=0.3, max_retries=5)
research_llm = create_react_agent(
    model=llm,
    tools=[scrape_tool, summarize_tool, search_tool],
    response_format=ResearchAgentOutput,
)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60))
def invoke_research_agent(brief: str) -> dict:
    return research_llm.invoke({"messages": [("human", brief)]})

def researcher_agent(state: AgentState) -> dict:
    """This agent takes a task from the planner and uses tools to gather findings relevant to it."""
    task = state["task"]
    current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

    brief = researcher_prompt.format_messages(
        current_date=current_time_str,
        title=task.title,
        instructions=task.instructions,
        guiding_questions="\n".join(f"- {q}" for q in task.guiding_questions),
    )

    result = invoke_research_agent(brief)
    items = result["structured_response"].items

    records = [
        ResearchRecord(agent_id=task.agent_id, task_title=task.title, **item.model_dump())
        for item in items
    ]
    print(records)
    return {"raw_research_data": records}