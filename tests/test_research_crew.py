import pytest
from agents.research_crew import run_research_crew


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_run_research_crew_returns_structured_claims():
    """Integration test for real MCP server, real LLM calls. 
    **Costly, not to be run by default.
    """
    results = run_research_crew(["What is the capital of France?"])
    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert result.sub_question == "What is the capital of France?"
    assert len(result.claims) > 0

    for claim in result.claims:
        assert isinstance(claim.text, str) and len(claim.text) > 0
        assert isinstance(claim.source_urls, list) and len(claim.source_urls) > 0
