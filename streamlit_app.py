"""
VakilAI — Streamlit Frontend
Connects to FastAPI backend at localhost:8000
True token streaming via LangServe /vakil/stream (SSE)
Redis-backed session history, UUID session isolation
"""

import uuid
import json
import requests
import streamlit as st

# ─── CONFIG ───────────────────────────────────────────────────────────
BACKEND_URL     = "http://localhost:8000"
ASK_ENDPOINT    = f"{BACKEND_URL}/ask"
INVOKE_ENDPOINT = f"{BACKEND_URL}/vakil/invoke"
STREAM_ENDPOINT = f"{BACKEND_URL}/vakil/stream_log"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

st.set_page_config(
    page_title="VakilAI — Indian Legal Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── STYLING ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Sans+3:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* Page background */
.stApp {
    background: #0f0f0e;
    color: #e8e4d9;
}

/* Header */
.vakil-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #2a2a28;
    margin-bottom: 2rem;
}

.vakil-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #c9a84c;
    margin: 0;
    letter-spacing: 0.02em;
}

.vakil-header p {
    font-size: 0.9rem;
    color: #6b6b65;
    margin: 0.4rem 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Chat messages */
.user-msg {
    background: #1a1a18;
    border: 1px solid #2a2a28;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #e8e4d9;
    font-size: 0.95rem;
    line-height: 1.6;
}

.assistant-msg {
    background: #161612;
    border: 1px solid #2e2b1f;
    border-left: 3px solid #c9a84c;
    border-radius: 4px 12px 12px 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #e8e4d9;
    font-size: 0.95rem;
    line-height: 1.7;
}

.msg-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    font-weight: 500;
}

.user-label { color: #6b6b65; }
.assistant-label { color: #c9a84c; }

/* Session badge */
.session-badge {
    display: inline-block;
    background: #1a1a18;
    border: 1px solid #2a2a28;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    color: #4a4a46;
    letter-spacing: 0.05em;
    font-family: monospace;
}

/* Status dot */
.status-online {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #4caf7d;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}

.status-offline {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #e24b4a;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}

/* Input area */
.stTextInput input {
    background: #1a1a18 !important;
    border: 1px solid #2a2a28 !important;
    border-radius: 8px !important;
    color: #e8e4d9 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
}

.stTextInput input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px #c9a84c30 !important;
}

/* Button */
.stButton button {
    background: #c9a84c !important;
    color: #0f0f0e !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
}

.stButton button:hover {
    opacity: 0.85 !important;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.72rem;
    color: #3a3a36;
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #1a1a18;
    margin-top: 2rem;
    letter-spacing: 0.03em;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 780px; }

/* Divider */
.msg-divider {
    border: none;
    border-top: 1px solid #1a1a18;
    margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE INIT ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_ok" not in st.session_state:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=2)
        st.session_state.backend_ok = r.status_code == 200
    except Exception:
        st.session_state.backend_ok = False

# ─── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="vakil-header">
    <h1>⚖️ VakilAI</h1>
    <p>Indian Legal Assistant</p>
</div>
""", unsafe_allow_html=True)

# Status + session row
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.backend_ok:
        st.markdown('<span class="status-online"></span><span style="font-size:0.75rem;color:#4a4a46;">Backend online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline"></span><span style="font-size:0.75rem;color:#6b3333;">Backend offline</span>', unsafe_allow_html=True)

with col2:
    short_id = st.session_state.session_id[:8]
    st.markdown(f'<div style="text-align:right"><span class="session-badge">session: {short_id}</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── CHAT HISTORY ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ─── INPUT ────────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            label="question",
            placeholder="Ask about Indian law — IPC, RTI, Constitution, Consumer Protection...",
            label_visibility="collapsed"
        )
    with col_btn:
        submitted = st.form_submit_button("Ask")

# ─── STREAMING GENERATOR ──────────────────────────────────────────────
# ─── STREAMING GENERATOR ──────────────────────────────────────────────
def stream_vakil(question: str, session_id: str):
    """
    Two-phase streaming strategy:

    Phase 1 — try /vakil/stream_log (SSE JSON patches from LangServe)
    Parses "streamed_output" patches to extract final answer string.
    Yields words one by one for live rendering.

    Phase 2 — fallback to /ask if stream_log fails
    Returns full answer, yields word by word to simulate streaming.
    """
    try:
        with requests.post(
            STREAM_ENDPOINT,
            json={"input": {"question": question, "session_id": session_id}},
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=60
        ) as response:

            if response.status_code != 200:
                raise Exception(f"stream_log returned {response.status_code}")

            final_output = ""

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data:"):
                    continue

                payload = decoded[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    break

                try:
                    patch = json.loads(payload)
                    # stream_log sends ops list — look for final output
                    ops = patch.get("ops", [])
                    for op in ops:
                        # Final streamed output lives at path "/streamed_output/-"
                        if op.get("path") == "/streamed_output/-":
                            chunk = op.get("value", "")
                            if isinstance(chunk, str) and chunk:
                                final_output += chunk
                        # Also catch final output directly
                        elif op.get("path") == "/final_output":
                            val = op.get("value", "")
                            if isinstance(val, str) and val:
                                final_output = val
                except (json.JSONDecodeError, AttributeError):
                    continue

            # Yield word by word for live rendering effect
            if final_output:
                words = final_output.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                return

    except Exception:
        pass  # fall through to /ask

    # ── Fallback: /ask endpoint ──
    try:
        r = requests.post(
            ASK_ENDPOINT,
            json={"question": question, "session_id": session_id},
            timeout=30
        )
        if r.status_code == 200:
            answer = r.json().get("answer", "Could not get answer.")
            words = answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
    except Exception as e:
        yield f"Error: {str(e)}"


# ─── HANDLE SUBMIT ────────────────────────────────────────────────────
if submitted and user_input.strip():

    if not st.session_state.backend_ok:
        st.error("Backend is offline. Start VakilAI server: `python3 main.py`")
        st.stop()

    question = user_input.strip()

    # Save + show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Stream assistant response token by token
    with st.chat_message("assistant"):
        try:
            # st.write_stream — calls the generator, renders each token live
            full_answer = st.write_stream(
                stream_vakil(question, st.session_state.session_id)
            )

            # Save complete answer to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer or ""
            })

        except requests.exceptions.ConnectionError:
            st.session_state.backend_ok = False
            st.error("Lost connection. Is `python3 main.py` running?")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Try again.")
        except Exception as e:
            # Fallback — hit /ask if stream fails
            try:
                r = requests.post(
                    ASK_ENDPOINT,
                    json={"question": question, "session_id": st.session_state.session_id},
                    timeout=30
                )
                if r.status_code == 200:
                    answer = r.json().get("answer", "No answer.")
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception:
                st.error(f"Error: {str(e)}")

# ─── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### VakilAI")
    st.markdown("---")

    st.markdown("**Session**")
    st.code(st.session_state.session_id, language=None)

    st.markdown("**Coverage**")
    st.markdown("""
    - Indian Penal Code 1860
    - Constitution of India
    - Consumer Protection Act 2019
    - RTI Act 2005
    """)

    st.markdown("---")

    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Start backend:**
    ```bash
    python3 main.py
    ```
    **Run frontend:**
    ```bash
    streamlit run streamlit_app.py
    ```
    """)

# ─── DISCLAIMER ───────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    VakilAI is for informational purposes only. Not a substitute for qualified legal advice.<br>
    For legal action, consult a licensed advocate.
</div>
""", unsafe_allow_html=True)
