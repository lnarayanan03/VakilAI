# graph.py
import os
from typing import TypedDict, Annotated, List
import operator

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_message_histories import RedisChatMessageHistory

from chain import ask_vakil, general_law_node_chain, web_search_chain
from schema import LegalAnswer

load_dotenv()

# ─── STATE DEFINITION ─────────────────────────────────────────────────
# Block 8: TypedDict — named, typed keys
# This is the message bus between all nodes
# Every node reads from and writes to this state
# Nodes return only the keys they changed — LangGraph merges automatically

class VakilAIState(TypedDict):
    question: str           # user's question — set once, never changes
    session_id: str         # user's session — for Redis memory isolation
    intent: str             # set by classifier node, read by router
    answer: dict            # set by legal_qa node, read by END (stored as model_dump())
    error: str              # set by error node if something fails
    last_question: str      # previous question for follow-up context

    # Block 8: Annotated with reducer
    # If multiple nodes write to messages simultaneously,
    # operator.add merges both lists instead of last-write-wins
    messages: Annotated[List[str], operator.add]

# ─── LLM FOR CLASSIFICATION ───────────────────────────────────────────
# Separate from chain.py's LLM — this one only classifies intent
# temperature=0 — deterministic routing, no creative surprises
# Block 7 concept: LLM as decision maker, not just text generator

classifier_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ─── NODE 1: CLASSIFIER ───────────────────────────────────────────────
# Block 8: plain Python function
# Receives full state, returns ONLY the keys it changed
# LangGraph merges returned dict into existing state automatically
#
# This is the first super-step — runs first on every invoke()
# Reads: state["question"]
# Writes: state["intent"]

def classify_intent(state: VakilAIState) -> dict:
    """
    Classifies user intent into one of four categories.
    Routes graph to appropriate handler.
    """
    question = state["question"]

    response = classifier_llm.invoke(
        f"""Classify this user message into exactly one category.

User message: "{question}"

Categories:
- legal_question: Any question about Indian law, rights, acts, sections,
  punishments, procedures, legal advice, or any matter governed by Indian
  statutes or the Constitution of India. Also includes questions about
  electoral reforms, delimitation, reservation policies, constitutional
  amendments, and any proposed or passed legislation in India.
- current_events: Questions about who currently holds a position
  (CM, PM, President, Governor, CJI), recent or latest election results,
  current government decisions, latest court judgments, new acts or bills
  introduced after 2023, any question needing up-to-date information about
  Indian politics or governance. Also includes questions using words:
  latest, recent, current, now, today, who won, who is the, what happened,
  2024, 2025, 2026
- greeting: Hello, hi, thanks, how are you, what can you do
- out_of_scope: Questions about non-Indian law, coding, weather,
  sports, or anything unrelated to Indian law

IMPORTANT: If the message is a short follow-up or contextual question that refers to a previous answer (e.g. 'tell me more', 'what about that', 'explain that', 'tell me about the first one', 'in that', 'elaborate', 'give me details') — classify it as legal_question. When in doubt, default to legal_question.

Reply with ONLY one word: legal_question, greeting, out_of_scope, or current_events"""
    )

    intent = response.content.strip().lower()

    # Sanitise — if LLM returns unexpected value, default to legal_question
    # Better to attempt an answer than to wrongly reject a legal question
    if intent not in ["legal_question", "greeting", "out_of_scope", "current_events"]:
        intent = "legal_question"
    
    return {"intent": intent}  # ← only return what changed

# ─── NODE 2: LEGAL QA ─────────────────────────────────────────────────
# Block 8: calls chain.py's ask_vakil
# This is the main work node — RAG + memory + structured output
# Reads: state["question"], state["session_id"]
# Writes: state["answer"]

def legal_qa_node(state: VakilAIState) -> dict:
    """
    Main RAG node. Retrieves context, generates structured answer.
    Uses chain.py which has full RAG + Redis memory wired in.
    """
    try:
        result = ask_vakil(
            question=state["question"],
            session_id=state["session_id"]
        )
        return {"answer": result.model_dump(), "last_question": state["question"]}
    except Exception as e:
        # If chain fails, return error state
        # Error node will handle it
        return {"error": str(e), "answer": None}

# ─── NODE 3: GREETING ─────────────────────────────────────────────────
# Block 8: simple deterministic node, no LLM call needed
# Reads: nothing from state
# Writes: state["answer"] as a simple LegalAnswer object

def greeting_node(state: VakilAIState) -> dict:
    """
    Handles greetings and farewells without burning LLM tokens.
    Returns a pre-built LegalAnswer with greeting or goodbye content.
    """
    question = state.get("question", "").lower().strip()
    farewell_words = ["bye", "goodbye", "see you", "take care",
                      "cya", "later", "goodnight", "good night",
                      "good bye", "tata", "ok bye", "okay bye"]
    is_farewell = any(w in question for w in farewell_words)

    if is_farewell:
        answer_text = (
            "Goodbye! Stay informed about your rights. "
            "VakilAI is always here when you need legal guidance. "
            "Take care! ⚖️"
        )
    else:
        answer_text = (
            "Hello! I am VakilAI, your Indian legal assistant. "
            "I can help you understand Indian laws including the IPC, "
            "RTI Act, Consumer Protection Act, and Constitution. "
            "Ask me any legal question!"
        )

    answer = LegalAnswer(
        answer=answer_text,
        applicable_law="N/A",
        section_numbers=[],
        confidence="high",
        found_in_context=True,
        disclaimer=""
    )
    return {"answer": answer.model_dump()}

# ─── NODE 4: OUT OF SCOPE ─────────────────────────────────────────────
# Block 8: handles questions outside Indian law domain
# Reads: nothing from state
# Writes: state["answer"]

def out_of_scope_node(state: VakilAIState) -> dict:
    """
    Handles questions outside VakilAI's domain gracefully.
    """
    answer = LegalAnswer(
        answer=(
            "I am VakilAI, specialised in Indian law only. "
            "I cannot help with this question. "
            "Please ask me about Indian laws, rights, or legal procedures."
        ),
        applicable_law="N/A",
        section_numbers=[],
        confidence="high",
        found_in_context=False,
        disclaimer=""
    )
    return {"answer": answer.model_dump()}

# ─── NODE 5: GENERAL LAW ─────────────────────────────────────────────
# Fallback when RAG finds no context ("could not find" in answer).
# Calls the general knowledge chain; converts OUT_OF_LEGAL_SCOPE to the
# standard out_of_scope LegalAnswer so downstream handling stays uniform.

def general_law_node(state: VakilAIState) -> dict:
    """
    Fallback to LLM general knowledge when RAG yields no context.
    """
    try:
        history_store = RedisChatMessageHistory(
            session_id=state["session_id"],
            url=os.getenv("REDIS_URL"),
            ttl=86400
        )
        history_messages = history_store.messages[-6:]  # last 3 exchanges
    except Exception:
        history_messages = []

    result = general_law_node_chain(state["question"], history=history_messages)

    if result.answer == "OUT_OF_LEGAL_SCOPE":
        result = LegalAnswer(
            answer=(
                "I am VakilAI, specialised in Indian law only. "
                "I cannot help with this question. "
                "Please ask me about Indian laws, rights, or legal procedures."
            ),
            applicable_law="N/A",
            section_numbers=[],
            confidence="high",
            found_in_context=False,
            disclaimer="",
        )

    return {"answer": result.model_dump()}


# ─── NODE 6: WEB SEARCH ──────────────────────────────────────────────
# Fires when LLM flags ##NEEDS_WEB_SEARCH## after RAG finds no context.
# Falls back to the existing LLM answer (flag stripped) if Tavily fails.

def web_search_node(state: VakilAIState) -> dict:
    """
    Fires when LLM flags ##NEEDS_WEB_SEARCH## in its answer.
    Combines LLM knowledge with live Tavily web search.
    """
    try:
        try:
            history_store = RedisChatMessageHistory(
                session_id=state["session_id"],
                url=os.getenv("REDIS_URL"),
                ttl=86400
            )
            history_messages = history_store.messages[-6:]  # last 3 exchanges
        except Exception:
            history_messages = []

        question = state["question"]
        last_question = state.get("last_question", "")

        # Detect follow-up questions
        followup_triggers = [
            "what about", "how about", "same for", "and now",
            "what about now", "now", "currently", "as of now",
            "in 2025", "in 2026", "2025", "2026", "latest",
            "recent", "update", "tell me more"
        ]
        is_followup = any(t in question.lower() for t in followup_triggers)

        # Build context-aware query
        if is_followup and last_question:
            search_question = f"{last_question} {question}"
        else:
            search_question = question

        result = web_search_chain(
            question=search_question,
            llm_answer=state["answer"].get("answer", "") if state.get("answer") else "",
            history=history_messages
        )
        return {"answer": result.model_dump()}
    except Exception as e:
        # Fallback — return existing LLM answer without web search
        if state.get("answer"):
            existing = state["answer"]
            existing["answer"] = existing.get("answer", "").replace("##NEEDS_WEB_SEARCH##", "").strip()
            existing["disclaimer"] = (
                "Web search unavailable. Answer from general legal knowledge only. "
                "Consult a qualified advocate."
            )
            return {"answer": existing}
        return {"error": str(e), "answer": None}


# ─── NODE 7: ERROR ────────────────────────────────────────────────────
# Block 8: fallback node when legal_qa_node fails
# Production systems always have a graceful error path

def error_node(state: VakilAIState) -> dict:
    """
    Graceful error handler. Never shows raw errors to users.
    """
    answer = LegalAnswer(
        answer=(
            "I encountered an issue processing your request. "
            "Please try again or rephrase your question."
        ),
        applicable_law="N/A",
        section_numbers=[],
        confidence="low",
        found_in_context=False,
        disclaimer="If this persists, please contact support."
    )
    return {"answer": answer.model_dump()}

# ─── ROUTING FUNCTION ─────────────────────────────────────────────────
# Block 8: conditional edge routing function
# Reads state["intent"] set by classifier node
# Returns the NAME of the next node to run
# LangGraph maps this name to the actual node

def route_intent(state: VakilAIState) -> str:
    """
    Routes to appropriate node based on classified intent.
    This is the conditional edge — graph-level if/else.
    """
    intent = state.get("intent", "legal_question")
    
    if intent == "legal_question":
        return "legal_qa"
    elif intent == "greeting":
        return "greeting"
    elif intent == "current_events":
        return "current_events"
    else:
        return "out_of_scope"

# ─── ALSO ROUTE AFTER LEGAL QA ────────────────────────────────────────
# If legal_qa_node set an error, go to error node
# Otherwise go to END

def route_after_qa(state: VakilAIState) -> str:
    if state.get("error"):
        return "error"
    answer = state.get("answer")
    if answer is None:
        return "error"
    # Check if RAG found nothing and answer signals web search needed
    if (
        not answer.get("found_in_context", True)
        and "##NEEDS_WEB_SEARCH##" in answer.get("answer", "")
    ):
        return "web_search"
    # Check if RAG found nothing and answer is the "could not find" default
    if (
        not answer.get("found_in_context", True)
        and any(phrase in answer.get("answer", "").lower() for phrase in [
            "could not find",
            "could not be found",
            "not found in the context",
            "unable to find",
            "no information found",
            "not in the context",
        ])
    ):
        return "general_law"
    return END

def route_after_general_law(state: VakilAIState) -> str:
    if state.get("error"):
        return "error"
    answer = state.get("answer")
    if answer is None:
        return "error"
    if "##NEEDS_WEB_SEARCH##" in answer.get("answer", ""):
        return "web_search"
    return END

# ─── BUILD THE GRAPH ──────────────────────────────────────────────────
# Block 8: StateGraph wires everything together
# compile() validates structure — catches dead ends, missing nodes

def build_graph():
    graph = StateGraph(VakilAIState)
    
    # Add all nodes
    graph.add_node("classifier",   classify_intent)
    graph.add_node("legal_qa",     legal_qa_node)
    graph.add_node("greeting",     greeting_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("error",        error_node)
    graph.add_node("general_law",  general_law_node)
    graph.add_node("web_search",   web_search_node)
    
    # Entry point — first node that runs
    graph.set_entry_point("classifier")
    
    # Conditional edges from classifier
    # Block 8: route_intent reads state["intent"], returns node name
    graph.add_conditional_edges(
        "classifier",
        route_intent,
        {
            "legal_qa":       "legal_qa",
            "greeting":       "greeting",
            "out_of_scope":   "out_of_scope",
            "current_events": "web_search",
        }
    )
    
    # After legal_qa — check for errors, web search flag, or missing context
    graph.add_conditional_edges(
        "legal_qa",
        route_after_qa,
        {
            "error":       "error",
            "general_law": "general_law",
            "web_search":  "web_search",
            END:           END,
        }
    )

    # Terminal nodes — all go to END
    graph.add_edge("greeting",     END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("error",        END)
    graph.add_conditional_edges(
        "general_law",
        route_after_general_law,
        {
            "web_search": "web_search",
            "error":      "error",
            END:          END,
        }
    )
    graph.add_edge("web_search",   END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# Build once at module load time
app = build_graph()

# ─── PUBLIC FUNCTION ──────────────────────────────────────────────────

def run_vakil(question: str, session_id: str) -> LegalAnswer:
    """
    Main entry point. Routes through LangGraph.
    """
    result = app.invoke(
        {
            "question":      question,
            "session_id":    session_id,
            "intent":        "",
            "answer":        None,
            "error":         "",
            "last_question": "",
            "messages":      [],
        },
        config={"configurable": {"thread_id": session_id}}
    )
    
    return LegalAnswer(**result["answer"]) if result["answer"] else None
