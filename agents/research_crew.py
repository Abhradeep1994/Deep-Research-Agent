from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from langchain_anthropic import ChatAnthropic
from graph.schemas import ResearchFindings, ResearchOutput

llm = LLM(model="anthropic/claude-sonnet-5")

SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_servers" / "research_server.py")

server_params = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
)

extraction_llm = ChatAnthropic(model="claude-sonnet-5")
structured_extractor = extraction_llm.with_structured_output(ResearchFindings)

EXTRACTION_PROMPT = """Extract distinct, atomic factual claims from the research answer below.
For each claim, list every source URL (mentioned in the text) that supports it.

Research answer:
{raw_answer}
"""


def extract_claims(raw_answer: str) -> ResearchFindings:
    return structured_extractor.invoke(EXTRACTION_PROMPT.format(raw_answer=raw_answer))


def run_research_crew(sub_questions: list[str]) -> list[ResearchOutput]:
    mcp_adapter = MCPServerAdapter(server_params)
    try:
        tools = mcp_adapter.tools

        researcher = Agent(
            role="Research Analyst",
            goal="Find accurate, well-sourced claims that answer a specific sub-question",
            backstory="An experienced analyst skilled at web research, who always separates "
                       "distinct facts and cites exactly which source(s) support each one.",
            tools=tools,
            llm=llm,
            verbose=True,
            max_iter=10,
        )
        tasks = [
            Task(
                description=(
                    f"Research this sub-question: {q}\n\n"
                    f"Use web_search to find candidate sources, then fetch_page to read their "
                    f"content. Write your findings as distinct, atomic factual claims — for each "
                    f"claim, cite every source URL that supports it (list multiple URLs if more "
                    f"than one source confirms it). Use no more than 5 total tool calls."
                ),
                expected_output=(
                    "A list of atomic factual claims, each followed by its supporting source URL(s)."
                ),
                agent=researcher,
            )
            for q in sub_questions
        ]
        crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=True)
        crew_output = crew.kickoff()

        results = []
        for q, task_output in zip(sub_questions, crew_output.tasks_output):
            findings = extract_claims(task_output.raw)
            results.append(ResearchOutput(sub_question=q, claims=findings.claims))
        return results
    finally:
        mcp_adapter.stop()
