from typing import TypedDict, List
from graph.schemas import ResearchOutput

class ResearchState(TypedDict):
    topic: str
    sub_questions: List[str]
    research_outputs: List[ResearchOutput]





