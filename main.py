# """Placeholder FastAPI backend. RAG, profile lookup and LLM get added later."""
# import uuid
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# app = FastAPI(title="SatyamAI API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class ChatRequest(BaseModel):
#     question: str
#     session_id: str | None = None


# @app.get("/health")
# def health():
#     return {"status": "ok", "index_loaded": False}


# @app.post("/chat")
# def chat(req: ChatRequest):
#     rid = f"req_{uuid.uuid4().hex[:8]}"
#     if any(w in req.question.lower() for w in ("kubernetes", "phone", "password")):
#         return {
#             "answer": "I don't have enough verified information to answer that.",
#             "evidence": [],
#             "confidence_status": "insufficient_evidence",
#             "request_id": rid,
#         }
#     return {
#         "answer": "Placeholder answer from the FastAPI backend.",
#         "evidence": [{"source": "placeholder.md", "location": "Section: Overview", "excerpt": "Sample excerpt."}],
#         "confidence_status": "placeholder",
#         "request_id": rid,
#     }







"""SatyamAI API: FastAPI backend with a LangChain + Chroma RAG pipeline."""
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()
log = logging.getLogger("satyamai")

# Load the RAG pipeline once at startup. If the index or API key is missing,
# the server still starts and /health reports index_loaded = false.
try:
    import rag
except Exception as e:
    rag = None
    log.error("RAG failed to load: %s (did you run `python ingest.py` and set GROQ_API_KEY?)", e)

NOT_FOUND = "I don't have enough verified information to answer that."
BLOCKED_WORDS = ("kubernetes", "phone", "password")   # fast refusal, no LLM call needed

def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)

# limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="SatyamAI API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=300)
    session_id: str | None = None


@app.get("/health")
def health():
    loaded = rag is not None and rag.index_size() > 0
    return {"status": "ok", "index_loaded": loaded}


@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, req: ChatRequest):
    rid = f"req_{uuid.uuid4().hex[:8]}"
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"

    def reply(answer, evidence, status):
        return {
            "answer": answer,
            "evidence": evidence,
            "confidence_status": status,
            "request_id": rid,
            "session_id": session_id,
        }

    if any(w in req.question.lower() for w in BLOCKED_WORDS):
        return reply(NOT_FOUND, [], "insufficient_evidence")

    if rag is None:
        return reply("The assistant is not ready yet. Please try again later.", [], "unavailable")

    try:
        result = rag.answer(req.question)
    except Exception:
        log.exception("chat failed (request %s)", rid)
        return reply("Something went wrong while answering. Please try again.", [], "error")

    return reply(result["answer"], result["evidence"], result["confidence_status"])

