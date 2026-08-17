import os
import json
from datetime import datetime

import wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.tools import Tool
from tenacity import retry, stop_after_attempt, wait_exponential

# Fix Wikimedia User-Agent block by setting a custom user agent
wikipedia.set_user_agent("MyResearchAgent/1.0 (contact: user@example.com)")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ---------------------------------------------------------------------------
# 1. Custom File Saving Function & Tool
# ---------------------------------------------------------------------------
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file.",
)


# ---------------------------------------------------------------------------
# 2. Web Search Tool
#
# Tavily is used when TAVILY_API_KEY is set: it's a licensed, agent-built
# search API with a real terms of service and paid tier, unlike DuckDuckGo's
# unofficial scraping libraries (ddgs), which its ToS explicitly prohibits
# for automated/commercial use. If no Tavily key is present yet (e.g. local
# dev before you've signed up), this automatically falls back to the
# DuckDuckGo-based tool from before so nothing breaks — but get a Tavily key
# (https://tavily.com) before using this commercially.
# ---------------------------------------------------------------------------
if TAVILY_API_KEY:
    from langchain_community.tools.tavily_search import TavilySearchResults

    _tavily = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)

    def _raw_search(query: str) -> str:
        results = _tavily.invoke(query)
        if isinstance(results, str):
            return results
        # Tavily returns a list of dicts like {"title", "url", "content"}
        formatted = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            formatted.append(f"{title} ({url}): {content}")
        return "\n\n".join(formatted) if formatted else "No results found."

    _SEARCH_BACKEND = "tavily"

else:
    from langchain_community.tools import DuckDuckGoSearchRun

    _ddg = DuckDuckGoSearchRun()

    def _raw_search(query: str) -> str:
        return _ddg.run(query)

    _SEARCH_BACKEND = "duckduckgo (fallback — set TAVILY_API_KEY for production use)"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=False,
)
def _search_with_retry(query: str) -> str:
    return _raw_search(query)


def safe_search(query: str) -> str:
    try:
        return _search_with_retry(query)
    except Exception as e:
        return (
            f"Search temporarily unavailable ({type(e).__name__}: {e}). "
            "Try relying on the wikipedia tool for this query, or rephrase "
            "and try the search tool again."
        )


search_tool = Tool(
    name="search",
    func=safe_search,
    description="Search the web for current information.",
    handle_tool_error=True,
)


# ---------------------------------------------------------------------------
# 3. Wikipedia Tool
# ---------------------------------------------------------------------------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    handle_tool_error=True,
)