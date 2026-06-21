from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from prompts import synthesizer_prompt

from llm import get_llm
from state import AgentState, ResearchRecord

llm = get_llm(temperature=0.2, max_retries=3)

def _is_low_confidence(confidence: str) -> bool:
    return (confidence or "").strip().lower() == "low"


def format_raw_data(raw_data: list[ResearchRecord]) -> str:
    """Group ResearchRecords by the task that produced them and render them
    as a prompt ready string. Pre-flags low-confidence and duplicate source
    records so the LLM doesn't have to spot them on its own.
    """
    grouped: dict[str, list[dict]] = {}
    seen_sources: set[tuple[str, str]] = set()

    for record in raw_data:
        data = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        grouped.setdefault(data.get("task_title", "Untitled Task"), []).append(data)

    formatted_str = ""
    for task_title, records in grouped.items():
        formatted_str += f"=== Task: {task_title} ===\n\n"
        for i, data in enumerate(records, start=1):
            content = data.get("content") or "N/A"
            confidence = data.get("confidence") or "N/A"
            source_key = (data.get("source", ""), data.get("title", ""))

            flags = []
            if _is_low_confidence(confidence):
                flags.append("LOW CONFIDENCE")
            if source_key in seen_sources:
                flags.append("DUPLICATE SOURCE")
            seen_sources.add(source_key)
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            formatted_str += f"--- Record {i}{flag_str} ---\n"
            formatted_str += f"Article Title: {data.get('title', 'N/A')}\n"
            formatted_str += f"Source URL/Location: {data.get('source', 'N/A')}\n"
            formatted_str += f"Confidence: {confidence}\n"
            formatted_str += f"Content: {content}\n\n"

    return formatted_str


def _coverage_note(state: AgentState, raw_research_data: list[ResearchRecord]) -> str:
    """Flag any planned research task that produced zero records, so a
    silently failed researcher sub-agent doesn't just disappear from the report."""
    planned = state.get("research_tasks", [])
    if not planned:
        return ""

    def as_dict(t):
        return t.model_dump() if hasattr(t, "model_dump") else t

    planned = [as_dict(t) for t in planned]
    produced_ids = {r.agent_id for r in raw_research_data}
    missing = [
        t.get("title") or t.get("agent_id", "unknown")
        for t in planned
        if t.get("agent_id") not in produced_ids
    ]

    if not missing:
        return ""
    return (
        "\nNote: the following planned research tasks returned no data and "
        f"should be flagged as gaps rather than omitted: {', '.join(missing)}\n"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def _invoke_synthesizer(chain, inputs: dict) -> str:
    response = chain.invoke(inputs)
    content = response.content
    return content if isinstance(content, str) else str(content)


def synthesizer_agent(state: AgentState) -> dict:
    """Takes raw gathered data from the state and compiles it into a single,
    human-readable Markdown research document."""

    clarified_query = state.get("clarified_query", "N/A")
    research_brief = state.get("research_brief", "N/A")
    raw_research_data = state.get("raw_research_data", [])

    if not raw_research_data:
        return {"compiled_research": "No research data was successfully gathered by the research agents."}

    raw_data_formatted = format_raw_data(raw_research_data)
    coverage_note = _coverage_note(state, raw_research_data)

    prompt = ChatPromptTemplate.from_template(synthesizer_prompt)
    chain = prompt | llm

    try:
        compiled = _invoke_synthesizer(
            chain,
            {
                "clarified_query": clarified_query,
                "research_brief": research_brief,
                "raw_data_formatted": raw_data_formatted,
                "coverage_note": coverage_note,
            },
        )
    except Exception as e:
        compiled = (
            f"Synthesis failed after multiple retries ({e}). "
            f"Raw research data is included below unsynthesized:\n\n{raw_data_formatted}"
        )

    return {"compiled_research": compiled}