from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from mcp_servers.research_server import web_search
from graph.schemas import Claim, ResearchOutput, ScoredClaim, VerifiedResearchOutput


class CorroborationCheck(BaseModel):
    corroborated: bool = Field(
        description="True only if at least one search result is from a DIFFERENT source "
                    "than the original and genuinely, specifically supports the claim"
    )
    supporting_url: str = Field(default="", description="URL of the corroborating result, if corroborated")
    reasoning: str = Field(description="Brief reasoning for the judgment")


corroboration_llm = ChatAnthropic(model="claude-sonnet-5")
structured_corroboration = corroboration_llm.with_structured_output(CorroborationCheck)

CORROBORATION_PROMPT = """You are fact-checking a single claim by searching for independent corroboration.

Claim: {claim_text}
Original source (already known — do not count this one as corroboration): {original_source}

Below are fresh web search results for this claim. Determine if any result is from a
DIFFERENT source than the original and genuinely, specifically supports the claim —
not just topically related.

Search results:
{search_results}
"""


def attempt_corroboration(claim: Claim) -> CorroborationCheck:
    results = web_search(claim.text)
    prompt = CORROBORATION_PROMPT.format(
        claim_text=claim.text,
        original_source=claim.source_urls[0] if claim.source_urls else "none",
        search_results=results,
    )
    return structured_corroboration.invoke(prompt)


def verify_claim(claim: Claim) -> ScoredClaim:
    if len(claim.source_urls) >= 2:
        return ScoredClaim(
            text=claim.text,
            source_urls=claim.source_urls,
            confidence=0.9,
            note=f"Pre-corroborated by {len(claim.source_urls)} independent sources during research.",
        )

    if len(claim.source_urls) == 1:
        check = attempt_corroboration(claim)
        if check.corroborated and check.supporting_url:
            return ScoredClaim(
                text=claim.text,
                source_urls=claim.source_urls + [check.supporting_url],
                confidence=0.85,
                note=f"Independently corroborated during verification: {check.reasoning}",
            )
        return ScoredClaim(
            text=claim.text,
            source_urls=claim.source_urls,
            confidence=0.4,
            note=f"Could not independently corroborate. {check.reasoning}",
        )

    return ScoredClaim(text=claim.text, source_urls=[], confidence=0.1, note="No sources provided.")


def verify_research_outputs(research_outputs: List[ResearchOutput]) -> List[VerifiedResearchOutput]:
    return [
        VerifiedResearchOutput(
            sub_question=ro.sub_question,
            claims=[verify_claim(c) for c in ro.claims],
        )
        for ro in research_outputs
    ]