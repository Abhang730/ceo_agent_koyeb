import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="CEO Agent Server (Groq-Powered)")

# Initialize Groq client using environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class TaskRequest(BaseModel):
    task: str

@app.get("/")
@app.get("/health")
def health():
    """Health check endpoint to verify server status."""
    return {
        "status": "ok",
        "service": "24/7 CEO Agent",
        "engine": "Groq Llama-3.3-70b-Versatile"
    }

@app.post("/delegate")
def delegate_task(req: TaskRequest):
    """CEO Logic: Evaluates user task, breaks it down, and creates worker instructions."""
    if not client:
        raise HTTPException(status_code=500, detail="Groq API key is missing on server.")
    
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task prompt cannot be empty.")

    system_prompt = (
        "You are the CEO Orchestrator AI. Your job is to analyze user requests and decompose them "
        "into structured, actionable tasks for two CLI workers:\n"
        "1. Freebuff CLI (for writing code, creating files, UI/web building).\n"
        "2. Hermes CLI (for deep research, web search, background execution, and state memory).\n\n"
        "Return a clear, structured JSON response with:\n"
        "- 'summary': High-level goal\n"
        "- 'hermes_task': Clear instruction for Hermes CLI (or null if not needed)\n"
        "- 'freebuff_task': Clear instruction for Freebuff CLI (or null if not needed)\n"
        "- 'review_criteria': How the CEO will verify success."
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.task}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        return {
            "status": "success",
            "ceo_plan": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")
