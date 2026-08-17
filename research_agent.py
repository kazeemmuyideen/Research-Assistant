import os
import re
import concurrent.futures

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


class MissingAPIKeyError(Exception):
    pass


def _friendly_error(e: Exception) -> str:
    """Turns known Gemini API errors into plain-English, actionable messages."""
    msg = str(e)

    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        if "free_tier" in msg or "FreeTier" in msg:
            return (
                "🚫 Gemini free-tier daily quota reached (20 requests/day for this model). "
                "Each research run uses several requests — one per agent iteration/tool call — "
                "so a handful of queries can use up the whole day's allowance.\n\n"
                "Options:\n"
                "- Wait for the quota to reset (resets daily; exact time depends on your account).\n"
                "- Lower 'Max agent iterations' in the sidebar so each run uses fewer requests.\n"
                "- Enable billing on your Google AI Studio / Gemini API project for much higher limits "
                "(see https://ai.google.dev/gemini-api/docs/rate-limits).\n"
                "- Repeated identical queries won't cost anything extra — they're cached for an hour."
            )
        return (
            "🚫 Gemini API rate limit hit (too many requests too quickly). "
            "This one is usually temporary — try again in a minute, or lower "
            "'Max agent iterations' to reduce requests per run."
        )

    if "API key not valid" in msg or "invalid api key" in msg.lower():
        return (
            "🔑 Gemini API key was rejected. Double-check GOOGLE_API_KEY in your .env file "
            "(or Streamlit Cloud secrets) is correct and active."
        )

    if _is_malformed_tool_call_error(e):
        return (
            "⚠️ The fallback model (Groq/Llama) generated a malformed tool call and couldn't "
            "recover, even after a retry. This is a known occasional quirk with smaller/faster "
            "models handling multiple tools plus a strict output format at once — it's not "
            "something wrong with your query. Try again, or narrow the question so the model "
            "needs fewer tool calls to answer it."
        )

    return f"Agent run failed: {e}"


def get_api_key() -> str | None:
    """Checks env vars first, then Streamlit secrets (for cloud deployment)."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def get_groq_api_key() -> str | None:
    """Optional fallback LLM provider — used automatically if Gemini fails
    (quota exhausted, outage, etc). Get a free key at https://console.groq.com."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def get_app_password() -> str | None:
    """Optional password gate, for when this is deployed publicly."""
    pw = os.getenv("APP_PASSWORD")
    if pw:
        return pw
    try:
        import streamlit as st

        return st.secrets.get("APP_PASSWORD")
    except Exception:
        return None


def _build_prompt_and_tools():
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use necessary tools.
                Wrap the output in this format and provide no other text\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    tools = [search_tool, wiki_tool, save_tool]
    return parser, prompt, tools


def _build_simple_prompt():
    """
    A deliberately looser prompt for the Groq fallback models: no strict JSON
    schema requirement alongside the tool-calling instructions. Smaller/faster
    models are noticeably more reliable at tool calling when they aren't
    simultaneously asked to hit an exact structured-output format — the JSON
    conversion happens afterward in a separate, tool-free step instead
    (see _build_reformat_chain), which is a much easier task on its own.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Use the available tools to research "
                "the user's query thoroughly, then write a clear, well-organized answer "
                "covering: what the topic/subject is, a detailed summary of findings, "
                "and which sources you drew from. Do not worry about any particular "
                "output format — just answer clearly in plain text.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


def _build_reformat_chain(llm, parser):
    """
    A separate, tool-free chain that takes the fallback agent's plain-language
    research findings and reformats them into the strict JSON schema. Splitting
    this out from the tool-calling step is the actual fix for the malformed
    tool call errors: a model asked to call tools AND hit an exact JSON schema
    in the same turn is doing two competing jobs at once. Reformatting plain
    text into JSON, with no tools in scope at all, is a much simpler task that
    small/fast models handle far more reliably.
    """
    reformat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a formatting assistant. Convert the research findings below "
                "into the exact JSON format specified. Output ONLY the JSON — no other "
                "text, no markdown fence.\n\n{format_instructions}",
            ),
            ("human", "Research findings:\n\n{raw_findings}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return reformat_prompt | llm


def _build_executor_from_llm(llm, prompt, tools, verbose, max_iterations, callbacks):
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        callbacks=callbacks or [],
        return_intermediate_steps=True,  # lets us reliably read which tools ran, regardless of final-answer format
    )


def build_agent_executor(verbose: bool = True, max_iterations: int = 3, callbacks=None):
    """
    Returns (agent_executor, parser, fallback_executors).

    fallback_executors is a list of separate agents built on Groq's free tier
    (across a few different models), used automatically by run_research() if
    the primary (Gemini) call fails with a quota/rate-limit error. They're
    tried in order — if one model produces a malformed tool call, the next
    model in the list is tried before giving up entirely. The list is empty
    if GROQ_API_KEY isn't set, in which case behavior is identical to before
    this fallback existed.

    Note: these are fully separate executors rather than llm.with_fallbacks(),
    because create_tool_calling_agent() calls llm.bind_tools() internally,
    which RunnableWithFallbacks doesn't support cleanly for tool-calling agents.
    """
    api_key = get_api_key()
    if not api_key:
        raise MissingAPIKeyError(
            "No GOOGLE_API_KEY (or GEMINI_API_KEY) found. Set it in a .env file "
            "locally, or in Streamlit Cloud's Secrets manager when deployed."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        max_retries=3,  # retries transient API errors (429/5xx/timeouts)
    )

    parser, prompt, tools = _build_prompt_and_tools()
    agent_executor = _build_executor_from_llm(llm, prompt, tools, verbose, max_iterations, callbacks)

    fallback_executors = []
    groq_key = get_groq_api_key()
    if groq_key:
        try:
            from langchain_groq import ChatGroq

            simple_prompt = _build_simple_prompt()

            # Tried in order. Override via GROQ_FALLBACK_MODELS (comma-separated)
            # in .env if you want a different lineup. Different model families
            # tend to fail on different queries, so chaining a few genuinely
            # helps rather than just retrying the same model repeatedly.
            default_models = "llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b"
            model_names = [
                m.strip()
                for m in os.getenv("GROQ_FALLBACK_MODELS", default_models).split(",")
                if m.strip()
            ]

            for model_name in model_names:
                try:
                    fallback_llm = ChatGroq(
                        model=model_name,
                        groq_api_key=groq_key,
                        max_retries=2,
                        temperature=0,  # deterministic output reduces malformed tool-call errors
                    )
                    # Fallback executors use the simpler prompt (no strict JSON
                    # demand) — the structured-output conversion happens
                    # afterward as a separate, tool-free step in run_research.
                    search_executor = _build_executor_from_llm(
                        fallback_llm, simple_prompt, tools, verbose, max_iterations, callbacks
                    )
                    reformat_chain = _build_reformat_chain(fallback_llm, parser)
                    fallback_executors.append(
                        {
                            "label": f"Groq ({model_name})",
                            "search_executor": search_executor,
                            "reformat_chain": reformat_chain,
                        }
                    )
                except Exception:
                    continue  # skip this model, try the next one
        except Exception:
            fallback_executors = []  # Groq package missing/misconfigured — degrade silently

    return agent_executor, parser, fallback_executors


def _extract_json_block(text: str) -> str:
    """
    Handles cases where the model wraps its JSON answer in a ```json fence
    or adds stray text around it, instead of returning bare JSON.
    """
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def _get_output_text(raw_response: dict) -> str:
    output = raw_response.get("output")
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return output[0].get("text", "")
    if isinstance(output, str):
        return output
    return ""


def _get_actual_tools_used(raw_response: dict) -> list:
    """
    Reads which tools genuinely ran from the agent's own execution log
    (intermediate_steps), rather than trusting the model's self-reported
    'tools_used' field in its JSON answer — which can hallucinate (e.g.
    listing Python libraries as "tools" when asked about Python, instead of
    reporting that it actually used the search/wikipedia tools).
    """
    steps = raw_response.get("intermediate_steps") if isinstance(raw_response, dict) else None
    if not steps:
        return []
    seen = []
    for action, _observation in steps:
        tool_name = getattr(action, "tool", None)
        if tool_name and tool_name not in seen:
            seen.append(tool_name)
    return seen


def _is_quota_or_rate_error(e: Exception) -> bool:
    msg = str(e)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "rate_limit" in msg.lower()


def _is_malformed_tool_call_error(e: Exception) -> bool:
    """
    Catches the handful of ways Groq/Llama's tool calling can go wrong under
    a multi-tool + strict-output-format prompt: outright API rejection
    ("Failed to call a function"), the tool name and JSON args getting glued
    together ("tool call validation failed... not in request.tools"), or a
    hallucinated tool name that doesn't exist in the tools list at all.
    """
    msg = str(e).lower()
    signatures = [
        "failed to call a function",
        "failed_generation",
        "tool call validation failed",
        "not in request.tools",
    ]
    return any(sig in msg for sig in signatures)


def _invoke_with_timeout(agent_executor, inputs, timeout_seconds, script_ctx):
    """Runs one agent in a background thread with a hard timeout and Streamlit
    context attached, so status-box callbacks work from the worker thread."""

    def _invoke():
        if script_ctx is not None:
            try:
                import threading
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                add_script_run_ctx(threading.current_thread(), script_ctx)
            except Exception:
                pass
        return agent_executor.invoke(inputs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        return future.result(timeout=timeout_seconds)


def run_research(
    query: str,
    agent_executor,
    parser,
    chat_history=None,
    timeout_seconds: int = 90,
    fallback_executors=None,
):
    """
    Runs the agent for a given query with a hard timeout, robust JSON parsing,
    and optional chat history for follow-up questions. If the primary run
    fails with a quota/rate-limit error, tries each executor in
    fallback_executors in turn (built automatically when GROQ_API_KEY is set
    across a few different Groq models) — if one model produces a malformed
    tool call, the next model is tried before giving up entirely.

    Returns:
    {
        "structured": ResearchResponse | None,
        "raw": <raw agent_executor output> | None,
        "error": str | None,
        "used_fallback": str | None,   # model description if a fallback answered, else None
    }
    """
    fallback_executors = fallback_executors or []
    inputs = {"query": query, "chat_history": chat_history or []}

    # Streamlit widgets (like the st.status box the callback handler writes to)
    # need to know which "script run" they belong to — capture that here on
    # the main thread since the agent itself runs in a worker thread.
    script_ctx = None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        script_ctx = get_script_run_ctx()
    except Exception:
        pass

    used_fallback = None
    raw_response = None

    try:
        raw_response = _invoke_with_timeout(agent_executor, inputs, timeout_seconds, script_ctx)
    except concurrent.futures.TimeoutError:
        return {
            "structured": None,
            "raw": None,
            "used_fallback": None,
            "error": (
                f"The agent took longer than {timeout_seconds}s and was stopped. "
                "If this is a well-documented topic, try increasing the timeout or "
                "adding context to narrow the search. If it's a specific private "
                "individual or obscure topic, a longer timeout may not help — search "
                "engines can only find what's actually indexed online, and no amount "
                "of retrying fixes thin or scattered web coverage."
            ),
        }
    except Exception as e:
        if fallback_executors and _is_quota_or_rate_error(e):
            last_fallback_error = e
            for idx, fb in enumerate(fallback_executors):
                try:
                    # Stage 1: let the model use tools freely, no JSON constraint.
                    search_raw = _invoke_with_timeout(
                        fb["search_executor"], inputs, timeout_seconds, script_ctx
                    )
                    plain_text = _get_output_text(search_raw)
                    if not plain_text and search_raw.get("intermediate_steps"):
                        # Some models leave "output" thin but did gather real
                        # findings via tools — fall back to summarizing those.
                        steps_text = "\n".join(
                            str(obs) for _, obs in search_raw["intermediate_steps"]
                        )
                        plain_text = steps_text[:4000]

                    if not plain_text.strip():
                        raise ValueError("Fallback model produced no findings to reformat.")

                    # Stage 2: separate, tool-free call that only reformats
                    # that plain-text answer into the strict JSON schema.
                    reformat_result = fb["reformat_chain"].invoke({"raw_findings": plain_text})
                    reformat_text = getattr(reformat_result, "content", None) or str(reformat_result)

                    raw_response = {
                        "output": [{"text": reformat_text}],
                        # keep the real tool-call log from stage 1 so tools_used
                        # can be set from what actually ran, not what the model claims
                        "intermediate_steps": search_raw.get("intermediate_steps", []),
                    }
                    used_fallback = fb["label"]
                    break
                except Exception as e_fb:
                    last_fallback_error = e_fb
                    continue  # this model failed at either stage — try the next one
            else:
                # every fallback model failed
                return {
                    "structured": None,
                    "raw": None,
                    "used_fallback": None,
                    "error": _friendly_error(last_fallback_error),
                }
        else:
            return {"structured": None, "raw": None, "used_fallback": None, "error": _friendly_error(e)}

    try:
        output_text = _get_output_text(raw_response)

        if "agent stopped due to" in output_text.lower() or "max iterations" in output_text.lower():
            return {
                "structured": None,
                "raw": raw_response,
                "used_fallback": used_fallback,
                "error": (
                    "The agent ran out of search attempts before settling on a confident "
                    "answer — this usually happens with ambiguous or hard-to-pin-down topics "
                    "(e.g. a name that matches several different people online). "
                    "Try increasing 'Max agent iterations' in the sidebar, or narrow the "
                    "question with more specific details."
                ),
            }

        json_text = _extract_json_block(output_text)
        structured_response = parser.parse(json_text)

        # Override the model's self-reported tools_used with what actually
        # ran, when we have a real execution log to check it against.
        actual_tools = _get_actual_tools_used(raw_response)
        if actual_tools:
            structured_response.tools_used = actual_tools

        return {
            "structured": structured_response,
            "raw": raw_response,
            "used_fallback": used_fallback,
            "error": None,
        }
    except Exception as e:
        return {
            "structured": None,
            "raw": raw_response,
            "used_fallback": used_fallback,
            "error": f"Error parsing response: {e}",
        }


def validate_sources(sources: list, timeout: float = 4.0) -> dict:
    """
    Best-effort HEAD request to check whether each source URL resolves.
    Returns True (reachable), False (unreachable), or None (not a URL / skipped).
    """
    results = {}
    for src in sources:
        url_match = re.search(r"https?://[^\s)]+", src)
        if url_match:
            url = url_match.group(0)
        else:
            # Catch bare domains written without a scheme, e.g. "python.org"
            # or "tiobe.com/tiobe-index/" — common when a model cites a site
            # by name rather than a full URL.
            bare_match = re.search(r"\b([a-z0-9-]+\.)+[a-z]{2,}(/[^\s)]*)?\b", src, re.IGNORECASE)
            if not bare_match:
                results[src] = None
                continue
            url = "https://" + bare_match.group(0)

        try:
            resp = requests.head(url, timeout=timeout, allow_redirects=True)
            results[src] = resp.status_code < 400
        except Exception:
            try:
                resp = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
                results[src] = resp.status_code < 400
            except Exception:
                results[src] = False
    return results


def to_chat_messages(history_pairs):
    """Converts a list of (query, ResearchResponse-or-None) into BaseMessage pairs."""
    messages = []
    for query, structured in history_pairs:
        messages.append(HumanMessage(content=query))
        if structured is not None:
            messages.append(AIMessage(content=structured.summary))
    return messages


if __name__ == "__main__":
    executor, parser, fallback_executors = build_agent_executor()
    q = input("What can I help you research: ")
    result = run_research(q, executor, parser, fallback_executors=fallback_executors)
    if result["error"]:
        print(result["error"], "Raw response -", result["raw"])
    else:
        print(result["structured"])