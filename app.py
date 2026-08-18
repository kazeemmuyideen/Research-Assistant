import streamlit as st

from research_agent import (
    build_agent_executor,
    run_research,
    run_deep_research,
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
    deep_research_mode = st.checkbox(
        "🔬 Deep research mode",
        value=False,
        help="Breaks the question into several sub-questions, researches each "
        "independently, then synthesizes one combined report. Slower and uses "
        "more API calls than a normal run, but produces more thorough coverage "
        "of broad topics.",
    )
    num_subquestions = 4
    if deep_research_mode:
        num_subquestions = st.slider("Number of sub-questions", 2, 6, 4)
        st.caption(
            f"Uses roughly {num_subquestions + 2} LLM calls per query "
            "(decompose + each sub-question + synthesis) — budget your daily "
            "quota accordingly."
        )

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


@st.cache_data(show_spinner=False, ttl=3600)
def cached_deep_run(query: str, num_subquestions: int, max_iterations: int, timeout_seconds: int):
    parser = st.session_state.parser
    # per-subquestion timeout — deep research does several sequential calls,
    # so each one gets a share of the overall timeout budget rather than the
    # full amount (otherwise total wait time could be timeout_seconds * N)
    per_sub_timeout = max(30, timeout_seconds // 2)
    return run_deep_research(
        query,
        parser,
        verbose=verbose,
        max_iterations=max_iterations,
        timeout_per_subquery=per_sub_timeout,
        num_subquestions=num_subquestions,
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

        status_box = st.status(
            "Running deep research..." if deep_research_mode else "Running agent...",
            expanded=True,
        )
        try:
            if deep_research_mode:
                # Deep research runs several sequential sub-agent calls plus a
                # synthesis step — live per-step streaming isn't wired through
                # the cache boundary here (same simplification the normal
                # cached_run already makes on a cache hit), so this just shows
                # a static status while it works rather than granular steps.
                status_box.write(
                    f"Breaking the question into up to {num_subquestions} sub-questions, "
                    "researching each, then synthesizing a combined report — this takes "
                    "noticeably longer than a normal run."
                )
                result = cached_deep_run(effective_query, num_subquestions, max_iterations, timeout_seconds)
                status_box.update(label="Done", state="complete", expanded=False)

                if result.get("raw") and result["raw"].get("sub_questions"):
                    with st.expander("🧩 Sub-questions researched", expanded=False):
                        for sq in result["raw"]["sub_questions"]:
                            st.markdown(f"- {sq}")

                try:
                    entry_id = history_store.save_entry(
                        f"[Deep research] {effective_query}", result.get("structured"), result.get("error")
                    )
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
                        st.markdown("**Summary:**")
                        st.write(r.summary)
                        if getattr(r, "full_report", None):
                            st.markdown("**Full Report:**")
                            st.write(r.full_report)

                if result.get("error"):
                    st.error(result["error"])

            else:
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
                        st.markdown("**Summary:**")
                        st.write(r.summary)
                        if getattr(r, "full_report", None):
                            st.markdown("**Full Report:**")
                            st.write(r.full_report)
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

                if row.get("full_report"):
                    st.markdown("**Full Report:**")
                    st.code(row["full_report"], language=None, wrap_lines=True)  # built-in copy button

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
                    md_parts = [f"# {row['topic']}", "", "## Summary", row["summary"] or ""]
                    if row.get("full_report"):
                        md_parts += ["", "## Full Report", row["full_report"]]
                    md_parts += [
                        "",
                        "## Sources",
                        "\n".join(f"- {s}" for s in row["sources"]),
                        "",
                        "## Tools used",
                        ", ".join(row["tools_used"]),
                    ]
                    st.download_button(
                        "⬇️ Markdown",
                        data="\n".join(md_parts),
                        file_name=f"{(row['topic'] or 'research')[:40].strip().replace(' ', '_')}.md",
                        mime="text/markdown",
                        key=f"md_{row['id']}",
                        use_container_width=True,
                    )
                with dl_col2:
                    pdf_bytes = build_research_pdf(
                        row["topic"] or "Research",
                        row["summary"],
                        row["sources"],
                        row["tools_used"],
                        full_report=row.get("full_report") or "",
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