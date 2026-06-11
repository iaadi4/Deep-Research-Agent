import os
from tavily import TavilyClient
from langchain_core.tools import tool
from pydantic import BaseModel, Field

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

@tool(args_schema=SearchInput)
def search_tool(query: str) -> str:
    """Searches the web and return top 5 results"""

    try:
        result = tavily_client.search(query, max_results=5)
    except Exception as e:
        return f"Failed to use search the web with query {query}, with error {e}"

    formatted = []

    for r in result.get("results", []):
        formatted.append(
            f"Title: {r.get('title', '')}\n"
            f"Content: {r.get('content', '')}\n"
            f"URL: {r.get('url', '')}"
        )
    
    return "\n\n".join(formatted)

