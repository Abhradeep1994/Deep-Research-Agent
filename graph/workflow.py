from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState

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
    return {'sub_questions': result.sub_questions}

builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_edge(START, "planner")
builder.add_edge("planner",END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"topic": "machine learning", "sub_questions":[]})
    for questions in result["sub_questions"]:
        print("-", questions)




