from typing import List
from pydantic import BaseModel, Field

class Claim(BaseModel):
    text: str = Field(description="A single factual claim relevant to the sub-question")
    source_urls: List[str] = Field(
        description = "URLs of sources that support this claim. Include every source that independetly corroborates it, not just the first one"
    )

class ResearchFindings(BaseModel):
    claims: List[Claim]

class ResearchOutput(BaseModel):
    sub_question: str
    claims: List[Claim]

class ScoredClaim(Claim):
    confidence: float = Field(description="0.0 to 1.0 confidence this claim is well-supported")
    note: str = Field(default="", description="How confidence was determined, or why it's low")


class VerifiedResearchOutput(BaseModel):
    sub_question: str
    claims: List[ScoredClaim]