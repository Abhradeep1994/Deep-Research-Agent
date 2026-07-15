from dotenv import load_dotenv
load_dotenv()

import argparse
from graph.workflow import graph


def run(topic: str) -> None:
    print(f"\nResearching: {topic}\n")
    print("This will take several minutes — planning, researching, verifying, "
          "and retrying if any claims come back under-supported.\n")

    result = graph.invoke({
        "topic": topic,
        "sub_questions": [],
        "research_outputs": [],
        "verified_outputs": [],
        "retry_count": 0,
        "final_report": "",
    })

    output_path = "output_report.md"
    with open(output_path, "w") as f:
        f.write(result["final_report"])

    print(f"\nDone. Retries used: {result['retry_count']}")
    print(f"Report written to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent — generate a cited research report on any topic."
    )
    parser.add_argument("topic", nargs="?", help="The research topic. If omitted, you'll be prompted.")
    args = parser.parse_args()

    topic = args.topic or input("What would you like researched? ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    run(topic)


if __name__ == "__main__":
    main()
