from typing import TypedDict, List
from graph.schemas import ResearchOutput, VerifiedResearchOutput

class ResearchState(TypedDict):
    topic: str
    sub_questions: List[str]
    research_outputs: List[ResearchOutput]
    verified_outputs: List[VerifiedResearchOutput]
    retry_count: int





