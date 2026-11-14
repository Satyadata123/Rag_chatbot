import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import DB_PATH, COLLECTION, PDF_PATH, get_embeddings

load_dotenv()

NOT_FOUND = "I don't have enough verified information to answer that."
SOURCE_NAME = os.path.basename(PDF_PATH)

# ---------- Vector store ----------
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=get_embeddings(),
    persist_directory=DB_PATH,
)

# ---------- LLM (verify current model names in Groq's docs) ----------
# _key = os.environ["GROQ_API_KEY"]
# primary = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=400, api_key=_key)
# backup = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=400, api_key=_key)
# llm = primary.with_fallbacks([backup])

# ---------- LLM ----------
_key = os.environ["GROQ_API_KEY"]
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
BACKUP_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

primary = ChatGroq(model=PRIMARY_MODEL, temperature=0.1, max_tokens=600, api_key=_key)
backup = ChatGroq(model=BACKUP_MODEL, temperature=0.1, max_tokens=600, api_key=_key)
llm = primary.with_fallbacks([backup])

# ---------- Prompt ----------
SYSTEM_PROMPT = f"""You are SatyamAI, the portfolio assistant for Satyam Singh.
Answer ONLY using the CONTEXT provided.
If the answer is not in the context, reply with exactly this sentence and nothing else:
"{NOT_FOUND}"
Never guess or invent numbers, dates, companies, skills or links.
The fraud detection and churn prediction projects have separate metrics. Never mix them.
Be concise. Use bullet points for lists. Do not mention page numbers or the word context."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
])

chain = prompt | llm | StrOutputParser()


def index_size() -> int:
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def retrieve(question: str, k: int = 5):
    q = question.lower()
    fraud = "fraud" in q or "ieee" in q
    churn = "churn" in q

    search_filter = None
    if fraud and not churn:
        search_filter = {"project": {"$in": ["fraud", "general"]}}
    elif churn and not fraud:
        search_filter = {"project": {"$in": ["churn", "general"]}}

    return vectorstore.similarity_search(question, k=k, filter=search_filter)


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata['heading']}]\n{d.page_content}" for d in docs)


def build_evidence(docs, limit: int = 3):
    evidence, seen = [], set()
    for d in docs:
        key = (d.metadata["page"], d.metadata["heading"])
        if key in seen:
            continue
        seen.add(key)
        excerpt = " ".join(d.page_content.split())   # collapse whitespace/newlines
        evidence.append({
            "source": SOURCE_NAME,
            "location": f"Page {key[0]} | Section: {key[1]}",
            "excerpt": excerpt[:300] + ("..." if len(excerpt) > 300 else ""),
        })
        if len(evidence) == limit:
            break
    return evidence


def answer(question: str) -> dict:
    docs = retrieve(question)
    text = chain.invoke({"context": format_docs(docs), "question": question}).strip()

    if not docs or NOT_FOUND.lower() in text.lower():
        return {"answer": NOT_FOUND, "evidence": [], "confidence_status": "insufficient_evidence"}

    # return {"answer": text, "evidence": build_evidence(docs), "confidence_status": "supported"}
    return {"answer": text, "evidence": build_evidence(docs), "confidence_status": "grounded"}