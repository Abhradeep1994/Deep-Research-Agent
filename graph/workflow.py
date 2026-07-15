from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState
from graph.schemas import VerifiedResearchOutput
from agents.research_crew import run_research_crew
from graph.verifier import verify_research_outputs

CONFIDENCE_THRESHOLD = 0.6
MAX_RETRIES = 2


class SubQuestions(BaseModel):
    sub_questions: List[str] = Field(
        description="3-5 specific, well-scoped, independently answerable "
                    "sub-questions that together thoroughly cover the research topic"
    )


llm = ChatAnthropic(model="claude-sonnet-5")
structured_llm = llm.with_structured_output(SubQuestions)


def planner_node(state: ResearchState) -> dict:
    prompt = (
        f"Break the following research topic into 3-5 specific, well-scoped "
        f"sub-questions that together would let someone thoroughly research it.\n\n"
        f"Topic: {state['topic']}"
    )
    result = structured_llm.invoke(prompt)
    return {"sub_questions": result.sub_questions}


def research_node(state: ResearchState) -> dict:
    research_outputs = run_research_crew(state["sub_questions"])
    return {"research_outputs": research_outputs}


def verifier_node(state: ResearchState) -> dict:
    verified_outputs = verify_research_outputs(state["research_outputs"])
    return {"verified_outputs": verified_outputs}


def get_low_confidence_sub_questions(verified_outputs: List[VerifiedResearchOutput]) -> List[str]:
    return [
        vo.sub_question
        for vo in verified_outputs
        if any(c.confidence < CONFIDENCE_THRESHOLD for c in vo.claims)
    ]


def route_after_verification(state: ResearchState) -> str:
    low_conf = get_low_confidence_sub_questions(state["verified_outputs"])
    if low_conf and state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "done"


def retry_research_node(state: ResearchState) -> dict:
    low_conf_vos = [
        vo for vo in state["verified_outputs"]
        if any(c.confidence < CONFIDENCE_THRESHOLD for c in vo.claims)
    ]
    low_conf_questions = [vo.sub_question for vo in low_conf_vos]
    feedback = {
        vo.sub_question: "; ".join(
            c.note for c in vo.claims if c.confidence < CONFIDENCE_THRESHOLD and c.note
        )
        for vo in low_conf_vos
    }

    new_outputs = run_research_crew(low_conf_questions, feedback=feedback)

    updated = {ro.sub_question: ro for ro in state["research_outputs"]}
    for new_ro in new_outputs:
        updated[new_ro.sub_question] = new_ro

    return {
        "research_outputs": list(updated.values()),
        "retry_count": state["retry_count"] + 1,
    }


builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("research_crew", research_node)
builder.add_node("verifier", verifier_node)
builder.add_node("retry_research", retry_research_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "research_crew")
builder.add_edge("research_crew", "verifier")
builder.add_conditional_edges(
    "verifier",
    route_after_verification,
    {"retry": "retry_research", "done": END},
)
builder.add_edge("retry_research", "verifier")

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "topic": "the impact of remote work on urban housing markets",
        "sub_questions": [],
        "research_outputs": [],
        "verified_outputs": [],
        "retry_count": 0,
    })
    print(f"\nRetries used: {result['retry_count']}\n")
    for vo in result["verified_outputs"]:
        print(f"Q: {vo.sub_question}")
        for c in vo.claims:
            print(f"  [{c.confidence:.2f}] {c.text}")
        print()