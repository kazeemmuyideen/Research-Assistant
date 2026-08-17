from langchain_core.callbacks import BaseCallbackHandler


class StreamlitStepHandler(BaseCallbackHandler):
    """Pushes short status lines to a Streamlit placeholder as the agent works."""

    def __init__(self, status_container):
        self.status_container = status_container
        self.steps = []

    def _push(self, line: str):
        self.steps.append(line)
        # st.status containers support .write for incremental lines
        self.status_container.write(line)

    def on_chain_start(self, serialized, inputs, **kwargs):
        self._push("🧠 Starting research...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        self._push(f"🔧 Using **{name}** — `{str(input_str)[:80]}`")

    def on_tool_end(self, output, **kwargs):
        self._push("✅ Tool finished")

    def on_tool_error(self, error, **kwargs):
        self._push(f"⚠️ Tool error: {error}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._push("💭 Thinking...")

    def on_agent_finish(self, finish, **kwargs):
        self._push("🏁 Finalizing answer...")