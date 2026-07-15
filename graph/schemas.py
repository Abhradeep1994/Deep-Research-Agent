import json
from typing import List
from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    text: str = Field(description="A single, atomic factual claim relevant to the sub-question")
    source_urls: List[str] = Field(
        description="URLs of sources that support this claim. Include every source that "
                    "independently corroborates it, not just the first one found."
    )


class ResearchFindings(BaseModel):
    """What the research agent produces for one sub-question."""
    claims: List[Claim]

    @field_validator("claims", mode="before")
    @classmethod
    def parse_claims_if_stringified(cls, v):
        # Defensive: LLM structured output for large nested arrays isn't perfectly
        # shape-reliable. Observed variants so far: (1) the array arrives as a
        # JSON-encoded string instead of a native list, (2) the array arrives
        # double-wrapped as {"claims": [...]} instead of just [...]. Normalize
        # both here rather than chase every possible shape via prompting.
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict) and "claims" in v:
            v = v["claims"]
        return v


class ResearchOutput(BaseModel):
    """A sub-question paired with its claims — what the graph state stores."""
    sub_question: str
    claims: List[Claim]


class ScoredClaim(Claim):
    confidence: float = Field(description="0.0 to 1.0 confidence this claim is well-supported")
    note: str = Field(default="", description="How confidence was determined, or why it's low")


class VerifiedResearchOutput(BaseModel):
    sub_question: str
    claims: List[ScoredClaim]
