from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

mcp = FastMCP("research-tools")


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about a query. Returns titles, URLs, and snippets for the top 5 results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Search error (no results available): {e}"

    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in results
    )


@mcp.tool()
def fetch_page(url: str) -> str:
    """Fetch a web page by URL and return its main readable text content (scripts, styles, and nav elements stripped). Use this after web_search to read the full content of a promising source."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:3000]


if __name__ == "__main__":
    mcp.run()
