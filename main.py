import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from graph import run_vakil


from typing_extensions import TypedDict

app = FastAPI(
    title="VakilAI",
    description="Indian Legal Assistant — powered by RAG + LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

class VakilRequest(BaseModel):
    question: str
    session_id: str = "default_session"

class PlaygroundInput(TypedDict):
    question: str
    session_id: str

vakil_runnable = RunnableLambda(
    lambda x: run_vakil(
        question=x["question"],
        session_id=x.get("session_id", "default_session")
    ).answer
).with_types(input_type=PlaygroundInput, output_type=str)

add_routes(
    app,
    vakil_runnable,
    path="/vakil",
    enabled_endpoints=["invoke", "batch", "playground", "stream_log"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "VakilAI"}

@app.post("/ask", response_model=dict)
def ask(request: VakilRequest):
    result = run_vakil(
        question=request.question,
        session_id=request.session_id
    )
    if result is None:
        return {
            "answer": "I encountered an issue processing your request. Please try again.",
            "applicable_law": "N/A",
            "section_numbers": [],
            "confidence": "low",
            "found_in_context": False,
            "disclaimer": "If this persists, please contact support."
        }
    return {
        "answer":           result.answer,
        "applicable_law":   result.applicable_law,
        "section_numbers":  result.section_numbers,
        "confidence":       result.confidence,
        "found_in_context": result.found_in_context,
        "disclaimer":       result.disclaimer,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
