from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from graph.schemas import ResearchFindings, ResearchOutput

llm = LLM(model="anthropic/claude-sonnet-5")
SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_servers" / "research_server.py")

server_params = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
)


def run_research_crew(sub_questions: list[str]) -> list[ResearchOutput]:
    mcp_adapter = MCPServerAdapter(server_params)
    try:
        tools = mcp_adapter.tools

        researcher = Agent(
            role="Research Analyst",
            goal="Find accurate, well-sourced claims that answer a specific sub-question",
            backstory="An experienced analyst skilled at web research, who always separates "
                       "distinct facts and tracks exactly which source(s) support each one.",
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
                    f"content. Break your findings into distinct, atomic factual claims. For each "
                    f"claim, list every source URL that supports it — if the same fact is confirmed "
                    f"by more than one independent source, list all of them. Use no more than 5 "
                    f"total tool calls."
                ),
                expected_output="A structured list of claims, each with its supporting source URL(s).",
                agent=researcher,
                output_pydantic=ResearchFindings,
            )
            for q in sub_questions
        ]
        crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=True)
        crew_output = crew.kickoff()

        return [
            ResearchOutput(sub_question=q, claims=task_output.pydantic.claims)
            for q, task_output in zip(sub_questions, crew_output.tasks_output)
        ]
    finally:
        mcp_adapter.stop()