from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from graph.workflow import graph

app = FastAPI(title="Deep Research Agent API")


class ResearchRequest(BaseModel):
    topic: str


class ResearchResponse(BaseModel):
    topic: str
    report: str
    retries_used: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <html>
      <head><title>Deep Research Agent</title></head>
      <body style="font-family: sans-serif; max-width: 640px; margin: 60px auto; line-height: 1.5;">
        <h1>Deep Research Agent</h1>
        <p>Enter a topic to generate a cited research report. This takes several minutes — planning, researching, verifying, and retrying if needed.</p>
        <form onsubmit="submitForm(event)">
          <input type="text" id="topic" style="width: 100%; padding: 10px; font-size: 16px;"
                 placeholder="e.g. the impact of remote work on urban housing markets" required>
          <br><br>
          <button type="submit" style="padding: 10px 20px; font-size: 16px;">Run Research</button>
        </form>
        <div id="status" style="margin-top: 20px; color: #666;"></div>
        <pre id="result" style="white-space: pre-wrap; margin-top: 20px;"></pre>
        <script>
          async function submitForm(e) {
            e.preventDefault();
            const topic = document.getElementById('topic').value;
            document.getElementById('status').textContent = 'Running... this can take several minutes. Do not close this tab.';
            document.getElementById('result').textContent = '';
            try {
              const res = await fetch('/research', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({topic})
              });
              const data = await res.json();
              document.getElementById('status').textContent =
                'Done. Retries used: ' + data.retries_used;
              document.getElementById('result').textContent = data.report;
            } catch (err) {
              document.getElementById('status').textContent = 'Error: ' + err;
            }
          }
        </script>
      </body>
    </html>
    """


@app.post("/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    result = graph.invoke({
        "topic": req.topic,
        "sub_questions": [],
        "research_outputs": [],
        "verified_outputs": [],
        "retry_count": 0,
        "final_report": "",
    })
    return ResearchResponse(
        topic=req.topic,
        report=result["final_report"],
        retries_used=result["retry_count"],
    )
