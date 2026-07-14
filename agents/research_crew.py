from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from tools.search_tool import web_search

llm = LLM(model="anthropic/claude-sonnet-5")

researcher = Agent(
    role = "Research Analyst",
    goal = "Find accurate, relevant, up to date information to answer a specific sub-question",
    backstory="An experienced analyst skilled at web research and synthesizing findings from multiple sources.",
    tools=[web_search],
    llm=llm,
    verbose=True,
    max_iter=10
)

def run_research_crew(sub_questions: list[str]) -> list[str]:
    tasks = [
        Task(
            description=(
                f"Research and answer this concisely, citing source URLs for every claim: {q}\n\n"
                f"Use no more than 3-4 targeted searches. Prioritize authoritative, recent sources "
                f"over exhaustive coverage."
            ),
            expected_output="A focused, well-organized answer (3-5 paragraphs) with inline source URLs.",
            agent=researcher,
        )
        for q in sub_questions
    ]
    crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=True)
    crew_output = crew.kickoff()
    return [t.raw for t in crew_output.tasks_output]