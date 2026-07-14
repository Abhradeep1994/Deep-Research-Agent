import pytest
from agents.research_crew import run_research_crew


@pytest.mark.slow
def test_run_research_crew_returns_findings():
    """Integration test — makes real LLM + web search calls.
    Costs a small amount of API credit; not run by default."""
    findings = run_research_crew(["What is the capital of France?"])
    assert isinstance(findings, list)
    assert len(findings) == 1
    assert isinstance(findings[0], str)
    assert len(findings[0]) > 0