from crewai.tools import tool
from ddgs import DDGS  

@tool("web_search")
def web_search(query:str)->str:
    """This tool is the web search tool which uses DDG search to find info"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in results
    )
