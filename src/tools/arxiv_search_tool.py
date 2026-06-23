import arxiv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tools.summarizer_tool import summarize_chain

class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query for arXiv papers. Supports field prefixes like 'au:' (author), 'ti:' (title), 'cat:' (category), and boolean operators (AND, OR, ANDNOT).")
    max_results: int = Field(default=5, description="Maximum number of papers to return (1-10)")


@tool(args_schema=ArxivSearchInput)
def arxiv_search_tool(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers, preprints, and scientific research.
    Use this tool when you need peer reviewed or scholarly sources, technical papers,
    state of the art research findings, or citations from the academic community.
    Returns titles, authors, abstracts, and PDF links for matching papers."""

    max_results = max(1, min(max_results, 10))

    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=3.0)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))
    except Exception as e:
        return f"Failed to search arXiv with query '{query}', error: {e}"

    if not results:
        return f"No arXiv papers found for query: '{query}'"

    formatted = []
    for paper in results:
        authors = ", ".join(a.name for a in paper.authors[:5])
        if len(paper.authors) > 5:
            authors += f" et al. ({len(paper.authors)} total)"

        categories = ", ".join(paper.categories) if paper.categories else "N/A"
        published = paper.published.strftime("%Y-%m-%d") if paper.published else "N/A"

        abstract = paper.summary or ""
        if abstract:
            try:
                summary_result = summarize_chain.invoke({
                    "tool_name": "arXiv",
                    "content": abstract,
                })
                abstract = summary_result.content
            except Exception:
                if len(abstract) > 800:
                    abstract = abstract[:800] + "... [TRUNCATED]"

        formatted.append(
            f"Title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Published: {published}\n"
            f"Categories: {categories}\n"
            f"Abstract: {abstract}\n"
            f"PDF: {paper.pdf_url}\n"
            f"arXiv URL: {paper.entry_id}"
        )

    return f"Found {len(results)} papers:\n\n" + "\n\n---\n\n".join(formatted)
