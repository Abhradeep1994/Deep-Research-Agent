import pytest
from graph.schemas import ScoredClaim, VerifiedResearchOutput
from graph.report import format_claims_for_prompt, assemble_report


def test_format_claims_for_prompt_splits_by_confidence():
    fake = [
        VerifiedResearchOutput(sub_question='Q1', claims=[
            ScoredClaim(text='Strong claim', source_urls=['url1', 'url2'], confidence=0.9, note='corroborated'),
            ScoredClaim(text='Weak claim', source_urls=['url3'], confidence=0.4, note='could not corroborate'),
        ]),
    ]
    lines, caveats = format_claims_for_prompt(fake)
    assert 'Strong claim' in lines
    assert 'Weak claim' not in lines
    assert any('Weak claim' in c for c in caveats)


def test_format_claims_for_prompt_deduplicates_across_sub_questions():
    fake = [
        VerifiedResearchOutput(sub_question='Q1', claims=[
            ScoredClaim(text='Duplicate claim', source_urls=['url1'], confidence=0.9, note=''),
        ]),
        VerifiedResearchOutput(sub_question='Q2', claims=[
            ScoredClaim(text='Duplicate claim', source_urls=['url1'], confidence=0.9, note=''),
        ]),
    ]
    lines, _ = format_claims_for_prompt(fake)
    assert lines.count('Duplicate claim') == 1


def test_format_claims_for_prompt_handles_empty_input():
    lines, caveats = format_claims_for_prompt([])
    assert lines == ""
    assert caveats == []


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_assemble_report_produces_markdown():
    """Integration test — real LLM call. Costs a small amount of API credit."""
    fake = [
        VerifiedResearchOutput(sub_question='What is the capital of France?', claims=[
            ScoredClaim(
                text='Paris is the capital of France.',
                source_urls=['https://en.wikipedia.org/wiki/Paris'],
                confidence=0.9,
                note='corroborated',
            ),
        ]),
    ]
    report = assemble_report("France's capital", fake)
    assert isinstance(report, str)
    assert len(report) > 0
    assert "Paris" in report
