# chain.py
import os
import re
import warnings
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from qdrant_client import QdrantClient

from schema import LegalAnswer

load_dotenv()

# ─── STEP 1: EMBEDDING MODEL ──────────────────────────────────────────
# Block 6: MUST be the same model used in ingest.py
# Different model = different vector space = garbage retrieval
# This is the MOST important rule in RAG

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ─── STEP 2: VECTOR STORE CONNECTION ──────────────────────────────────
# Block 6: connect to existing Qdrant collection
# ingest.py created this — we just connect to it
# No re-embedding here — just reading what's already stored

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION", "vakil_legal"),
    embedding=embedder,
)

# ─── STEP 3: RETRIEVER ────────────────────────────────────────────────
# Block 6: score_threshold over fixed k
# Only returns chunks with similarity > 0.3
# Dynamic — might return 2 chunks or 8 depending on query
# Better than hardcoded k=4 which always returns exactly 4
# even if 3 of them are irrelevant

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.3,  # lower threshold = more recall for legal Q&A
        "k": 10,                   # max 6 chunks
    }
)

# ─── STEP 4: FORMAT DOCS ──────────────────────────────────────────────
# Block 6: converts List[Document] → string for {context} slot
# Critical: {context} is a STRING not a list
# Without this, prompt crashes with template error

def format_docs(docs: list) -> str:
    if not docs:
        return "No relevant legal context found."
    
    return "\n\n".join(
        f"[{doc.metadata.get('act', 'Unknown Act')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

# ─── STEP 5: PROMPT ───────────────────────────────────────────────────
# Block 3: ChatPromptTemplate with all 4 components
# system     → who VakilAI is + strict context grounding
# history    → conversation memory (optional=True for first turn)
# context    → retrieved legal chunks (formatted string)
# question   → user's question

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are VakilAI, an Indian legal assistant.\n"
     "Answer questions using ONLY the legal context provided below.\n"
     "Always cite the exact Act name and Section number.\n"
     "If the answer is in the context above, you MUST set found_in_context to true.\n"
     "If the answer is not in the context at all, set found_in_context to false and say you could not find it.\n"
     "Never hallucinate legal information — Indian law is precise.\n\n"
     "IMPORTANT: found_in_context must be JSON boolean true or false (not the string 'True' or 'False').\n\n"
     "Legal Context:\n{context}"
    ),
    # Block 3: MessagesPlaceholder with optional=True
    # optional=True → no crash on first turn when history=[]
    # Injects List[HumanMessage, AIMessage, ...] from Redis
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}")
])

# ─── STEP 6: LLM WITH STRUCTURED OUTPUT ───────────────────────────────
# Block 5: .with_structured_output() uses function calling protocol
# LangChain converts LegalAnswer → JSON Schema → sends as tool definition
# API forces tool_choice=required → LLM cannot return free text
# Returns typed LegalAnswer object, not a string
# temperature=0 → deterministic extraction, no creative surprises

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,   # Block 5: always 0 for structured extraction
)

structured_llm = llm.with_structured_output(LegalAnswer)

# ─── STEP 7: RAG CHAIN ────────────────────────────────────────────────
# Block 2: LCEL pipe — definition only, nothing runs yet
# 
# RunnableParallel (the dict):
#   "context"  branch: retriever → format_docs (does real work)
#   "question" branch: RunnablePassthrough (forwards input unchanged)
#   Both run simultaneously via asyncio.gather()
#
# Then sequential: prompt → structured_llm
# No StrOutputParser needed — structured_llm already returns LegalAnswer

rag_chain = (
    {
        "context":  RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | structured_llm
)

# ─── PUBLIC FUNCTION ──────────────────────────────────────────────────
# Called by graph.py — clean interface

def ask_vakil(question: str, session_id: str) -> LegalAnswer:
    """
    Main entry point for VakilAI.
    Loads Redis history, injects into RAG chain, saves exchange back.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Load conversation history from Redis
        try:
            history_store = RedisChatMessageHistory(
                session_id=session_id,
                url=os.getenv("REDIS_URL"),
                ttl=86400
            )
            history_messages = history_store.messages[-6:]  # last 3 exchanges
        except Exception:
            history_messages = []

        # Invoke RAG chain with history
        result = rag_chain.invoke({
            "question": question,
            "history": history_messages
        })

        # Save exchange to Redis
        try:
            history_store.add_user_message(question)
            history_store.add_ai_message(result.answer)
        except Exception:
            pass  # Redis failure never breaks the main flow

        return result


def general_law_node_chain(question: str, history: list = []) -> LegalAnswer:
    """
    Fallback node for questions not found in the RAG index.
    Answers from LLM general knowledge with a strict law-only prompt.
    Returns LegalAnswer(answer="OUT_OF_LEGAL_SCOPE") for non-legal questions.
    """
    system_prompt = (
        "You are VakilAI, an Indian legal assistant with deep knowledge of "
        "Indian law. Answer ONLY questions related to Indian law, legal "
        "procedures, rights, acts, sections, and court judgments.\n\n"
        "STRICT RULES:\n"
        "1. If the question is completely unrelated to Indian law (coding, "
        "   weather, sports, foreign law, entertainment) → respond with "
        "   exactly: OUT_OF_LEGAL_SCOPE\n\n"
        "2. If the question is about Indian law but requires information "
        "   after 2023 (new acts, recent amendments, recent court judgments, "
        "   new government policies, bills introduced after 2023) → answer "
        "   what you know up to 2023, then append exactly: "
        "   ##NEEDS_WEB_SEARCH##\n\n"
        "3. If the question is about Indian law and you can answer it "
        "   fully from your knowledge → answer completely.\n\n"
        "Always mention the relevant Act name and section if applicable.\n"
        "Always end with: This answer is from general legal knowledge. "
        "For authoritative legal advice, consult a qualified advocate."
    )

    history_dicts = [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant",
         "content": msg.content}
        for msg in history
    ]

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            *history_dicts,
            {"role": "user",   "content": question},
        ])
        answer_text = response.content.strip()
    except Exception:
        answer_text = (
            "I could not process this question. "
            "Please rephrase and try again. "
            "This answer is from general legal knowledge. "
            "For authoritative legal advice, consult a qualified advocate."
        )

    if answer_text == "OUT_OF_LEGAL_SCOPE":
        return LegalAnswer(
            answer="OUT_OF_LEGAL_SCOPE",
            applicable_law="N/A",
            section_numbers=[],
            confidence="low",
            found_in_context=False,
            disclaimer="",
        )

    # Extract the first Act name from the answer if present
    match = re.search(r'[A-Z][A-Za-z\s]*Act(?:\s+\d{4})?', answer_text)
    applicable_law = match.group(0).strip() if match else "General Legal Knowledge"

    return LegalAnswer(
        answer=answer_text,
        applicable_law=applicable_law,
        section_numbers=[],
        confidence="medium",
        found_in_context=False,
        disclaimer=(
            "This answer is from general legal knowledge, not indexed documents. "
            "Consult a qualified advocate."
        ),
    )


def web_search_chain(question: str, llm_answer: str, history: list = []) -> LegalAnswer:
    """
    Triggered when LLM flags ##NEEDS_WEB_SEARCH##.
    Searches web for current info, combines with LLM knowledge,
    returns complete answer.
    """
    # Strip the flag from existing answer
    clean_llm_answer = llm_answer.replace("##NEEDS_WEB_SEARCH##", "").strip()

    # Search web
    search = TavilySearch(
        max_results=4,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )
    raw = search.invoke({"query": f"India law {question}"})

    # Tavily returns dict with "results" key containing the list
    result_list = raw.get("results", []) if isinstance(raw, dict) else []

    web_context = "\n\n".join([
        f"[Source: {r.get('url', 'Unknown')}]\n{r.get('content', '')}"
        for r in result_list
    ]) if result_list else "No web results found."

    # Combine LLM knowledge + web results → final answer
    web_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are VakilAI, an Indian legal assistant.\n"
         "You have two sources of information:\n"
         "1. Your own legal knowledge (reliable till 2023)\n"
         "2. Recent web search results (may contain post-2023 info)\n\n"
         "Using BOTH sources, give a complete and accurate answer.\n\n"
         "CRITICAL RULES:\n"
         "1. For election results, court verdicts, government decisions — "
         "   ONLY state what is CONFIRMED in the web search results. "
         "   Do NOT declare a winner or outcome unless the web results "
         "   explicitly say the result is FINAL and DECLARED.\n"
         "2. If results are still being counted or trends are preliminary — "
         "   clearly say 'Results are still being counted as of [date]. "
         "   Current trends show X but final results are not yet declared.'\n"
         "3. Never extrapolate or predict outcomes from partial data.\n"
         "4. If web results are conflicting — mention both and say "
         "   results are unclear.\n"
         "5. Only answer questions related to Indian law, governance, "
         "   and constitutional matters.\n"
         "6. If web results are completely irrelevant to Indian law — "
         "   ignore them and answer from your own knowledge only.\n"
         "7. Always cite source URLs from web results when used.\n\n"
         "Your existing knowledge on this topic:\n{llm_answer}\n\n"
         "Recent web search results:\n{web_context}"
        ),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{question}")
    ])

    web_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_web_llm = web_llm.with_structured_output(LegalAnswer)

    web_chain = web_prompt | structured_web_llm

    result = web_chain.invoke({
        "question": question,
        "llm_answer": clean_llm_answer,
        "web_context": web_context,
        "history": history
    })

    # Override fields for web-sourced answers
    result.found_in_context = False
    result.confidence = "medium"
    result.disclaimer = (
        "This answer combines general legal knowledge and recent web search results. "
        "For authoritative legal advice, consult a qualified advocate."
    )
    return result
