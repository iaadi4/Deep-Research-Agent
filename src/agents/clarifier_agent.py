from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field
from pydantic import BaseModel
from state import AgentState
from llm import get_llm
from prompts.clarifier_prompt import clarifier_prompt
from tools.tavily_search_tool import search_tool
import datetime


search_query_llm = get_llm(temperature=0.0)
llm = get_llm(temperature=0.7)

class ClarifierOutput(BaseModel):
    is_query_clear: bool = Field(
        description="Set to true if query is clear enough to start researching. Set to false if need clarification"
    )
    response_content: str = Field(
        description="If is_query_clear is False, put 1-3 clarification questions. Else put the polished detailed RESEARCH_OBJECTIVE here."
    )


clarifier_llm = clarifier_prompt | llm.with_structured_output(ClarifierOutput)


def _derive_search_query(user_queries: list) -> str:
    """
    Uses a lightweight LLM call to synthesize an optimal Tavily search query
    from the full conversation history.

    Why LLM instead of heuristics:
    - Heuristics (char-length thresholds, string slicing) break on short
      confirmations ("yes", "option A"), indirect references ("the second one"),
      multilingual input, or topic pivots mid-conversation.
    - The LLM understands *intent* across the entire thread — it can resolve
      "yes" against the previous AI question, combine multiple partial turns
      into one coherent topic, and always produce a search-optimised query
      regardless of conversation shape.

    Prompt is intentionally minimal and zero-temperature to keep latency low.
    Output is a single raw search string — no JSON, no formatting.
    """
    if not user_queries:
        return None

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in user_queries
    )

    synthesis_prompt = (
        "You are a search query synthesizer. Given a conversation thread between a user and an AI research assistant, "
        "your sole job is to output the single best web search query that would retrieve real-world news or factual context "
        "about the core topic being discussed.\n\n"
        "Rules:\n"
        "- Output ONLY the raw search query string. No explanation, no punctuation, no quotes.\n"
        "- Resolve short replies ('yes', 'no', 'the second one', 'correct') against the prior AI question to infer intent.\n"
        "- Prefer named entities (people, places, events, organisations) over vague terms.\n"
        "- Bias toward recency: include a year or 'recent' only if the topic is time-sensitive.\n"
        "- Max 12 words.\n\n"
        f"Conversation:\n{history_text}\n\n"
        "Search query:"
    )

    response = search_query_llm.invoke(synthesis_prompt)
    query = response.content.strip().strip('"').strip("'")
    return query


def clarifier_agent(state: AgentState) -> dict:
    """This node clarifies the scope of research regarding user queries"""

    search_query = _derive_search_query(state["user_queries"])

    if search_query is None:
        search_context = "No grounding context available — conversation has not started yet."
    else:
        try:
            search_context = search_tool.invoke(search_query)
        except Exception as e:
            search_context = f"No recent search data found. Error: {e}"

    current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

    grounding_message = AIMessage(
        content=(
            f"[GROUNDING CONTEXT — VERIFIED LIVE WEB DATA as of {current_time_str}]:\n"
            f"{search_context}\n"
            "[END GROUNDING CONTEXT]"
        )
    )

    result = clarifier_llm.invoke({
        "current_date": current_time_str,
        "conversation_history": state["user_queries"] + [grounding_message],
    })

    state_update = {
        "is_query_clear": result.is_query_clear,
        "clarified_query": result.response_content,
    }

    if not result.is_query_clear:
        state_update["user_queries"] = [AIMessage(content=result.response_content)]
        print(result.response_content)

    return state_update


def should_move_to_planner(state: AgentState) -> str:
    """Decides whether to move to the planner agent or loop back for clarification"""
    if state["is_query_clear"] is True:
        return "continue"
    else:
        return "loop"


def user_input_node(state: AgentState) -> dict:
    """This node takes user input for research"""
    query = input("Enter: ")
    return {"user_queries": [HumanMessage(content=query)]}