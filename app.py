import streamlit as st

from research_agent import (
    build_agent_executor,
    run_research,
    validate_sources,
    to_chat_messages,
    get_api_key,
    get_app_password,
    get_groq_api_key,
    MissingAPIKeyError,
    _friendly_error,
)
from streamlit_callbacks import StreamlitStepHandler
from pdf_export import build_research_pdf
import history_store

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="wide")
history_store.init_db()

# ---------------------------------------------------------------------------
# Password gate (only active if APP_PASSWORD is set — safe to leave unset locally)
# ---------------------------------------------------------------------------
app_password = get_app_password()
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Research Assistant")
        pw_input = st.text_input("Enter password", type="password")
        if st.button("Unlock"):
            if pw_input == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    max_iterations = st.slider("Max agent iterations", 1, 10, 5)
    timeout_seconds = st.slider("Timeout (seconds)", 30, 300, 120, step=15)
    verbose = st.checkbox("Verbose agent logs (terminal)", value=True)
    debug_mode = st.checkbox("Debug mode (show raw agent output)", value=False)
    check_sources = st.checkbox("Validate source links", value=True)
    use_followups = st.checkbox("Multi-turn (use chat history as context)", value=False)

    st.divider()
    st.caption(
        "Set `GOOGLE_API_KEY` in a `.env` file locally, or in Streamlit Cloud's "
        "Secrets manager when deployed. Optionally set `APP_PASSWORD` to gate access."
    )

    fallback_configured = get_groq_api_key() is not None
    st.caption(
        f"Fallback LLMs (Groq): {'✅ configured — tries multiple models in sequence' if fallback_configured else '⚪ not set — add `GROQ_API_KEY` for automatic failover if Gemini quota is hit'}"
    )

    if st.button("🔄 Rebuild agent"):
        st.session_state.pop("agent_executor", None)
        st.session_state.pop("parser", None)
        st.session_state.pop("fallback_executors", None)
        st.rerun()

    st.divider()
    st.subheader("🗂️ History")
    st.caption(f"Storage backend: **{history_store.backend_name()}**")
    if st.button("🗑️ Clear saved history", use_container_width=True):
        history_store.clear_history()
        st.session_state.pop("run_results", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Build agent (cached in session so we don't rebuild every rerun)
# ---------------------------------------------------------------------------
if "agent_executor" not in st.session_state:
    try:
        with st.spinner("Setting up the research agent..."):
            executor, parser, fallback_executors = build_agent_executor(
                verbose=verbose, max_iterations=max_iterations
            )
            st.session_state.agent_executor = executor
            st.session_state.parser = parser
            st.session_state.fallback_executors = fallback_executors
    except MissingAPIKeyError as e:
        st.error(f"⚠️ {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to set up the agent: {e}")
        st.stop()

# run_results: id -> {"structured": ResearchResponse|None, "error": str|None}
# kept separately from history_store rows so we can attach live objects (for PDF etc.)
if "run_results" not in st.session_state:
    st.session_state.run_results = {}

if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

# ---------------------------------------------------------------------------
# Cached wrapper — avoids re-spending API credits on an identical query
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_run(query: str, max_iterations: int, timeout_seconds: int, chat_history_key: str = ""):
    # chat_history_key exists purely so identical follow-up context busts the cache
    # correctly; the real chat_history objects live in session state below.
    executor, parser = st.session_state.agent_executor, st.session_state.parser
    fallback_executors = st.session_state.get("fallback_executors", [])
    chat_history = st.session_state.get("_pending_chat_history", [])
    return run_research(
        query,
        executor,
        parser,
        chat_history=chat_history,
        timeout_seconds=timeout_seconds,
        fallback_executors=fallback_executors,
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🔎 Research Assistant")
st.write(
    "Ask a research question — the agent will search, pull from Wikipedia, and can save results."
)

query = st.text_area(
    "What can I help you research?",
    value=st.session_state.prefill_query,
    height=100,
    placeholder="e.g. Latest advances in solid-state batteries",
    key="query_box",
)

with st.expander("➕ Add context to narrow an ambiguous query (optional)"):
    st.caption(
        "Useful for names, small companies, or anything that could match multiple "
        "results — e.g. a location, occupation, or organization you know is correct."
    )
    extra_context = st.text_input(
        "Additional context",
        placeholder="e.g. Nigerian software developer, based in Lagos",
        key="extra_context_box",
    )
    st.caption(
        "Note: for private individuals with common names and little web presence, "
        "no amount of context fully guarantees a confident match — search results "
        "are limited by what's actually indexed online."
    )

st.session_state.prefill_query = ""  # only prefill once, on the run right after a click

col1, col2 = st.columns([1, 5])
with col1:
    run_clicked = st.button("Run research", type="primary", use_container_width=True)

if run_clicked:
    if not query.strip():
        st.warning("Please enter a research question first.")
    else:
        # Fold optional disambiguating context directly into the query sent
        # to the agent — this is what actually helps it converge instead of
        # thrashing through iterations on an ambiguous name/topic.
        extra_context = st.session_state.get("extra_context_box", "").strip()
        effective_query = f"{query.strip()} (context: {extra_context})" if extra_context else query.strip()

        status_box = st.status("Running agent...", expanded=True)
        try:
            chat_history = []
            if use_followups:
                from langchain_core.messages import HumanMessage, AIMessage

                for e in reversed(history_store.load_history(limit=6)):
                    if e["summary"]:
                        chat_history.append(HumanMessage(content=e["query"]))
                        chat_history.append(AIMessage(content=e["summary"]))

            # Attach live-step streaming to the agent for this run. This is set
            # on the session's executor instance directly (not inside the cached
            # function) so the handler still fires even though the *result* gets
            # cached — on a cache hit, no new agent call happens so no new steps
            # will stream, which is expected: it means we skipped an API call.
            step_handler = StreamlitStepHandler(status_box)
            st.session_state.agent_executor.callbacks = [step_handler]
            st.session_state["_pending_chat_history"] = chat_history

            result = cached_run(effective_query, max_iterations, timeout_seconds, chat_history_key=str(chat_history))
            status_box.update(label="Done", state="complete", expanded=False)

            if result.get("used_fallback"):
                st.info(f"ℹ️ Gemini was unavailable for this run — used {result['used_fallback']} instead.")

            # Saving to history is a separate concern from the research run
            # itself — if the store is briefly unreachable, don't make it
            # look like the whole research run failed.
            try:
                entry_id = history_store.save_entry(effective_query, result.get("structured"), result.get("error"))
            except Exception:
                entry_id = None

            if entry_id is not None:
                st.session_state.run_results[entry_id] = result
                st.session_state["latest_entry_id"] = entry_id
            else:
                st.warning("Research completed, but couldn't save it to history right now.")
                if result.get("structured"):
                    r = result["structured"]
                    st.markdown(f"**Topic:** {r.topic}")
                    st.write(r.summary)
        except Exception as e:
            status_box.update(label="Failed", state="error")
            st.error(_friendly_error(e))

# ---------------------------------------------------------------------------
# Results — pulled from persistent SQLite history so it survives reloads
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Results")

try:
    history_rows = history_store.load_history(limit=25)
except Exception as e:
    st.warning(
        f"Couldn't reach the history store right now ({type(e).__name__}). "
        "Your results still ran — this just affects the saved history view. Try refreshing."
    )
    history_rows = []

if not history_rows:
    st.info("Your research results will show up here once you run a query.")
else:
    latest_id = st.session_state.get("latest_entry_id")
    for row in history_rows:
        is_latest = row["id"] == latest_id
        with st.expander(f"🔹 {row['query']}  ·  {row['timestamp']}", expanded=is_latest):
            if row["error"]:
                st.error(row["error"])
                if debug_mode:
                    live = st.session_state.run_results.get(row["id"])
                    if live and live.get("raw"):
                        st.json(live["raw"])
            else:
                st.markdown(f"**Topic:** {row['topic']}")
                st.markdown("**Summary:**")
                st.code(row["summary"], language=None, wrap_lines=True)  # built-in copy button

                if row["sources"]:
                    st.markdown("**Sources:**")
                    link_status = {}
                    if check_sources:
                        with st.spinner("Checking source links..."):
                            link_status = validate_sources(row["sources"])
                    for s in row["sources"]:
                        if check_sources:
                            ok = link_status.get(s)
                            icon = "✅" if ok else ("❔" if ok is None else "⚠️")
                            st.markdown(f"{icon} {s}")
                        else:
                            st.markdown(f"- {s}")

                if row["tools_used"]:
                    st.markdown("**Tools used:** " + ", ".join(row["tools_used"]))

                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    st.download_button(
                        "⬇️ Markdown",
                        data=(
                            f"# {row['topic']}\n\n{row['summary']}\n\n"
                            f"## Sources\n" + "\n".join(f"- {s}" for s in row["sources"])
                            + f"\n\n## Tools used\n{', '.join(row['tools_used'])}"
                        ),
                        file_name=f"{(row['topic'] or 'research')[:40].strip().replace(' ', '_')}.md",
                        mime="text/markdown",
                        key=f"md_{row['id']}",
                        use_container_width=True,
                    )
                with dl_col2:
                    pdf_bytes = build_research_pdf(
                        row["topic"] or "Research", row["summary"], row["sources"], row["tools_used"]
                    )
                    st.download_button(
                        "⬇️ PDF",
                        data=pdf_bytes,
                        file_name=f"{(row['topic'] or 'research')[:40].strip().replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{row['id']}",
                        use_container_width=True,
                    )
                with dl_col3:
                    if st.button("🔁 Re-run this query", key=f"rerun_{row['id']}", use_container_width=True):
                        st.session_state.prefill_query = row["query"]
                        st.rerun()

            if st.button("🗑️ Delete this entry", key=f"del_{row['id']}"):
                history_store.delete_entry(row["id"])
                st.rerun()