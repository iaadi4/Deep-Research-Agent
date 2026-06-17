import os
from pydantic import Field
from pydantic import BaseModel
from firecrawl import Firecrawl
from langchain_core.tools import tool

MAX_CHARS = 4000  # max chars returned to agent after scraping website content

firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
firecrawl_client = Firecrawl(api_key=firecrawl_api_key)

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(description="url of website to be scraped")


@tool(args_schema=ScrapeWebsiteInput)
def scrape_tool(url: str) -> dict:
    """Scrape website and return cleaned data"""
    
    try:
        result = firecrawl_client.scrape(
            url,
            formats = ["markdown"],
            only_main_content = True,
            max_age = 86400000
        )
    except Exception as e:
        return {
            "error": f"failed to scrape url: {url}, with error {e}",
            "url": url
        }

    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "dict"):
        data = result.dict()
    elif isinstance(result, dict):
        data = result.get("data", result)
    else:
        data = {}

    metadata = data.get("metadata") or {}
    raw_content = data.get("markdown") or ""

    if len(raw_content) > MAX_CHARS:
        content = raw_content[:MAX_CHARS] + "\n\n [CONTENT TRUNCATED FOR LENGTH]"
    else:
        content = raw_content

    return {
        "title": metadata.get("title", ""),
        "url": metadata.get("sourceURL", url),
        "description": metadata.get("description", ""),
        "content": content
    }

