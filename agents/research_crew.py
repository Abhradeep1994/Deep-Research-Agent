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
)

def run_research_crew(sub_questions: list[str]) -> list[str]:
    tasks = [
        Task(
            description=f"Research and answer this thoroughly, citing source URLs for every claim: {q}",
            expected_output="A well-organized answer with inline source URLs.",
            agent=researcher,
        )
        for q in sub_questions
    ]
    crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=True)
    crew_output = crew.kickoff()
    return [t.raw for t in crew_output.tasks_output]