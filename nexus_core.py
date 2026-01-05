import streamlit as st
import pandas as pd
import os
import time
import uuid
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Database Imports
from nexus_db import init_db, save_message, load_history, clear_session, save_setting, load_setting, get_all_sessions
from themes import THEMES, inject_theme_css

# --- 1. CONFIGURATION & KEYS ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

raw_groq = st.secrets.get("GROQ_API_KEYS", "")
raw_tavily = st.secrets.get("TAVILY_API_KEYS", "")

GROQ_KEYS = [k.strip() for k in raw_groq.split(",") if k.strip()]
TAVILY_KEYS = [k.strip() for k in raw_tavily.split(",") if k.strip()]

if not GROQ_KEYS or not TAVILY_KEYS:
    st.error("⚠️ System Halted: Missing API Keys in Secrets.")
    st.stop()

# Key Rotation State
if "groq_idx" not in st.session_state: st.session_state.groq_idx = 0
if "tavily_idx" not in st.session_state: st.session_state.tavily_idx = 0


def get_current_keys():
    g_key = GROQ_KEYS[st.session_state.groq_idx % len(GROQ_KEYS)]
    t_key = TAVILY_KEYS[st.session_state.tavily_idx % len(TAVILY_KEYS)]
    os.environ["GROQ_API_KEY"] = g_key
    os.environ["TAVILY_API_KEY"] = t_key
    return g_key, t_key


def rotate_groq_key():
    st.session_state.groq_idx = (st.session_state.groq_idx + 1) % len(GROQ_KEYS)
    new_key = GROQ_KEYS[st.session_state.groq_idx]
    os.environ["GROQ_API_KEY"] = new_key
    return new_key


get_current_keys()

MODEL_SMART = "llama-3.3-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"


# --- 2. DATA ENGINE (WITH TEXT SUCCESS TRIGGER) ---
class DataEngine:
    def __init__(self):
        self.scope = {
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "st": st
        }
        self.df = None
        self.column_str = ""

    def load_file(self, uploaded_file):
        try:
            name = uploaded_file.name
            if name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif 'xls' in name:
                    self.df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    self.df = pd.read_json(uploaded_file)

                self.column_str = ", ".join(list(self.df.columns))
                self.scope["df"] = self.df
                return f"✅ Data Loaded: {len(self.df)} rows. Columns: {self.column_str}"
            elif name.endswith(('.txt', '.py', '.md', '.log', '.yaml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()
                self.scope["file_content"] = self.file_content
                return f"✅ Text Loaded: {len(self.file_content)} chars."
            else:
                return f"⚠️ Binary file '{name}' (Limited Access)."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _heal_code(self, code: str) -> str:
        """Autocorrects code to prevent common crashes."""
        if self.df is None: return code

        # 1. FIX: Numeric Columns Only for Correlation
        if ".corr()" in code and "numeric_only" not in code:
            code = code.replace(".corr()", ".select_dtypes(include=['number']).corr()")

        # 2. FIX: Column Name Typos (Case-Insensitive)
        real_cols = list(self.df.columns)
        col_map = {c.lower(): c for c in real_cols}
        pattern = r"df\[['\"](.*?)['\"]\]"

        def replace_match(match):
            col_name = match.group(1)
            lower_name = col_name.lower()
            if col_name not in real_cols and lower_name in col_map:
                correct_name = col_map[lower_name]
                return f"df['{correct_name}']"
            return match.group(0)

        healed_code = re.sub(pattern, replace_match, code)
        return healed_code

    def run_python_analysis(self, code: str):
        # Apply Auto-Healing
        code = self._heal_code(code)

        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            plt.clf()
            exec(code, self.scope)
            result = redirected_output.getvalue()

            # 1. Check for Charts
            if plt.get_fignums():
                st.pyplot(plt)
                plt.clf()
                return f"Output:\n{result}\n[CHART GENERATED]"

            # 2. Check for Text Output (THE FIX FOR 'HOW MANY' QUESTIONS)
            if result and len(result.strip()) > 0:
                return f"Output:\n{result}\n[ANALYSIS COMPLETE]"

            return f"Output:\n{result}" if result else "Code executed successfully."

        except KeyError as e:
            cols = self.column_str if self.column_str else "Data Not Loaded"
            return f"❌ Column Error: {str(e)}\n💡 AVAILABLE COLUMNS: {cols}"

        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout


# --- 3. STATE INIT ---
if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
engine = st.session_state.data_engine

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
current_sess = st.session_state.current_session_id

init_db()

# --- 4. SIDEBAR ---
current_theme = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme)
theme_data = THEMES.get(current_theme, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")
    st.caption(f"Keys: Groq({st.session_state.groq_idx + 1}) | Tavily({st.session_state.tavily_idx + 1})")

    uploaded_file = st.file_uploader("📂 Upload File", type=None)
    if uploaded_file:
        status = engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

    if st.button("🧹 Clear Plots"):
        plt.clf()
        st.pyplot(plt)

    st.divider()
    if st.button("➕ New Chat"):
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()
    if st.button("🗑️ Clear Chat"):
        clear_session(current_sess)
        st.rerun()

    st.markdown("### 🕒 History")
    for s in get_all_sessions()[:5]:
        if st.button(f"{s}", key=s):
            st.session_state.current_session_id = s
            st.rerun()


# --- 5. AGENT SETUP ---
class PythonInput(BaseModel):
    code: str = Field(description="Python code. Use 'df'.")


def python_analysis_tool(code: str):
    return engine.run_python_analysis(code)


has_data = "df" in engine.scope or "file_content" in engine.scope


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def get_fresh_tools():
    get_current_keys()
    tools = [TavilySearchResults(max_results=2)]
    if has_data:
        tools.append(StructuredTool.from_function(
            func=python_analysis_tool,
            name="python_analysis",
            description="Run Python code.",
            args_schema=PythonInput
        ))
    return tools


def agent_node(state):
    primary_model = MODEL_SMART if has_data else MODEL_FAST
    models = [primary_model, MODEL_FAST] if primary_model != MODEL_FAST else [MODEL_FAST]

    last_error = None
    for model in models:
        for i in range(len(GROQ_KEYS)):
            try:
                tools = get_fresh_tools()
                key = os.environ["GROQ_API_KEY"]
                llm = ChatGroq(model=model, temperature=0.1, api_key=key).bind_tools(tools)
                return {"messages": [llm.invoke(state["messages"])]}
            except Exception as e:
                last_error = e
                if "429" in str(e): rotate_groq_key(); continue
                break

    return {"messages": [AIMessage(content=f"❌ System Exhausted. Error: {str(last_error)}")], "final": True}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(get_fresh_tools()))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- 6. CHAT UI ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

history = load_history(current_sess)
current_messages = []
for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]):
        st.markdown(msg["content"])
    current_messages.append(
        HumanMessage(content=msg["content"]) if role == "user" else AIMessage(content=msg["content"]))

if prompt := st.chat_input("Enter command..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    # REFRESH COLUMNS
    if engine.df is not None and not engine.column_str:
        engine.column_str = ", ".join(list(engine.df.columns))

    system_text = "You are Nexus."
    if has_data:
        system_text += f"""
        [DATA MODE]
        1. Variable 'df' is loaded.
        2. VALID COLUMNS: [{engine.column_str}]
        3. FOR PLOTS: `plt.figure()` -> `plt.plot()` -> NO `plt.show()`.
        4. FOR NUMBERS/COUNTS: Use `print()`. Example: `print(df['Promotion'].value_counts())`.
        5. STOPPING: If tool output says "[ANALYSIS COMPLETE]" or "[CHART GENERATED]", STOP.
        """
    else:
        system_text += " If no file, use 'tavily'."

    current_messages.append(SystemMessage(content=system_text))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        try:
            final_resp = ""
            for event in app.stream({"messages": current_messages}, config={"recursion_limit": 50},
                                    stream_mode="values"):
                msg = event["messages"][-1]
                if isinstance(msg, ToolMessage): status_box.write(f"⚙️ Output: {msg.content[:200]}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for t in msg.tool_calls: status_box.write(f"⚙️ Calling: `{t['name']}`")
                if isinstance(msg, AIMessage) and msg.content:
                    final_resp = msg.content
                    st.markdown(final_resp)

            if final_resp:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_resp)
            else:
                st.error("No response.")
        except Exception as e:
            status_box.update(label="Error", state="error");
            st.error(f"Error: {e}")