from tools.search_tool import web_search
from tools import search_tool
from crewai.tools import tool
from ddgs import DDGS

def test_web_search_returns_results():
    """Integration test to hit the real DuckDuckGo search API."""
    result = web_search.func("Python programming language")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "No results found" not in result


def test_web_search_handles_no_results(monkeypatch):
    """Unit test for external dependency so this test is fast,
    deterministic, and doesn't depend on network access."""

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, query, max_results=5):
            return []

    monkeypatch.setattr(search_tool, "DDGS", FakeDDGS)
    result = web_search.func("anything")
    assert result == "No results found."