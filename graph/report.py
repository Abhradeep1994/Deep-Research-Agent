from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple
from langchain_anthropic import ChatAnthropic
from graph.schemas import VerifiedResearchOutput

CONFIDENCE_PUBLISH_THRESHOLD = 0.6

report_llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=8192)

REPORT_PROMPT = """You are writing a polished, cited research report on the topic below, \
based ONLY on the verified claims provided. Do not add outside information or invent facts.

Topic: {topic}

Verified claims, each with a confidence score and source URLs:

{claims_block}

Write a well-organized Markdown report:
- A brief introduction framing the topic
- 3-5 thematic sections (not one section per sub-question) that synthesize related claims \
into flowing prose, grouped by theme rather than by which sub-question produced them
- Every factual statement should cite its source URL inline, e.g. ([source](URL))
- A brief concluding synthesis
- If the same fact appears more than once in the claims list, mention it once, not repeatedly

Do not include a separate references section — sources are cited inline only.
"""


def format_claims_for_prompt(verified_outputs: List[VerifiedResearchOutput]) -> Tuple[str, List[str]]:
    """Splits claims into (confident claims formatted for the report prompt, caveat lines
    for low-confidence claims), deduplicating by exact claim text."""
    lines = []
    caveats = []
    seen = set()
    for vo in verified_outputs:
        for c in vo.claims:
            if c.text in seen:
                continue
            seen.add(c.text)
            if c.confidence >= CONFIDENCE_PUBLISH_THRESHOLD:
                lines.append(f"- [{c.confidence:.2f}] {c.text} (Sources: {', '.join(c.source_urls)})")
            else:
                caveats.append(f"- {c.text} (confidence: {c.confidence:.2f} — {c.note})")
    return "\n".join(lines), caveats


def extract_text_content(content) -> str:
    """Claude's response .content can be a plain string, or (with adaptive thinking
    models) a list of content blocks including a 'thinking' block alongside the real
    'text' block. Normalize to plain text either way."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def assemble_report(topic: str, verified_outputs: List[VerifiedResearchOutput]) -> str:
    claims_block, caveats = format_claims_for_prompt(verified_outputs)
    prompt = REPORT_PROMPT.format(topic=topic, claims_block=claims_block)
    raw_response = report_llm.invoke(prompt)
    report_body = extract_text_content(raw_response.content)

    if caveats:
        report_body += (
            "\n\n---\n\n## Lower-Confidence Findings (Not Independently Corroborated)\n\n"
            "The following claims appeared during research but could not be independently "
            "verified above our confidence threshold. Included for transparency, not asserted "
            "as established fact:\n\n" + "\n".join(caveats)
        )

    return report_body
