import streamlit as st
import pandas as pd
import os
import time
import uuid
import sys
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

# --- 1. CONFIGURATION & KEY ROTATION SYSTEM ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

# LOAD KEYS (Comma-Separated)
raw_groq = st.secrets.get("GROQ_API_KEYS", "")
raw_tavily = st.secrets.get("TAVILY_API_KEYS", "")

GROQ_KEYS = [k.strip() for k in raw_groq.split(",") if k.strip()]
TAVILY_KEYS = [k.strip() for k in raw_tavily.split(",") if k.strip()]

if not GROQ_KEYS or not TAVILY_KEYS:
    st.error("⚠️ System Halted: Missing API Keys in Secrets.")
    st.info("Please paste the 'GROQ_API_KEYS' and 'TAVILY_API_KEYS' block into your Streamlit Secrets.")
    st.stop()

# STATE: Track which key is currently active
if "groq_idx" not in st.session_state: st.session_state.groq_idx = 0
if "tavily_idx" not in st.session_state: st.session_state.tavily_idx = 0


def get_current_keys():
    """Sets the environment variables to the CURRENT active key index."""
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


def rotate_tavily_key():
    st.session_state.tavily_idx = (st.session_state.tavily_idx + 1) % len(TAVILY_KEYS)
    new_key = TAVILY_KEYS[st.session_state.tavily_idx]
    os.environ["TAVILY_API_KEY"] = new_key
    return new_key


# Initialize Keys
get_current_keys()

# MODELS - Removed dead models
MODEL_SMART = "llama-3.3-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"


# --- 2. DATA ENGINE ---
class DataEngine:
    def __init__(self):
        self.scope = {
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "st": st
        }
        self.df = None
        self.file_content = None

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
                self.scope["df"] = self.df
                return f"✅ Data Loaded: {len(self.df)} rows."
            elif name.endswith(('.txt', '.py', '.md', '.log', '.yaml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()
                self.scope["file_content"] = self.file_content
                return f"✅ Text Loaded: {len(self.file_content)} chars."
            else:
                return f"⚠️ Binary file '{name}' (Limited Access)."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def run_python_analysis(self, code: str):
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            plt.clf()  # Clear previous plots
            exec(code, self.scope)
            result = redirected_output.getvalue()

            if plt.get_fignums():
                st.pyplot(plt)
                plt.clf()
                # 🛑 FORCE STOP SIGNAL
                return f"Output:\n{result}\n[SYSTEM MSG: Chart Successfully Rendered. TASK COMPLETE. STOP NOW.]"

            return f"Output:\n{result}" if result else "Code executed successfully."
        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout


# --- 3. STATE INIT ---
if "data_engine" not in st.session_state:
    st.session_state.data_engine = DataEngine()
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
    st.caption(f"🔑 **Groq Key:** {st.session_state.groq_idx + 1}/{len(GROQ_KEYS)}")
    st.caption(f"🔑 **Tavily Key:** {st.session_state.tavily_idx + 1}/{len(TAVILY_KEYS)}")

    uploaded_file = st.file_uploader("📂 Upload File", type=None)
    if uploaded_file:
        status = engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

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


# --- 5. TOOLS ---
class PythonInput(BaseModel):
    code: str = Field(description="Python code to run. Use 'df'.")


def python_analysis_tool(code: str):
    return engine.run_python_analysis(code)


has_data = "df" in engine.scope or "file_content" in engine.scope


# --- 6. AGENT NODE (Failover + Multi-Key) ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def get_fresh_tools():
    """Refreshes tools to pick up any environment variable changes"""
    get_current_keys()
    t_tool = TavilySearchResults(max_results=2)
    tool_list = [t_tool]
    if has_data:
        tool_list.append(StructuredTool.from_function(
            func=python_analysis_tool,
            name="python_analysis",
            description="Run Python code.",
            args_schema=PythonInput
        ))
    return tool_list


def agent_node(state):
    primary_model = MODEL_SMART if has_data else MODEL_FAST
    models_to_try = [primary_model, MODEL_FAST]
    if primary_model == MODEL_FAST: models_to_try = [MODEL_FAST]

    last_error = None

    for model_name in models_to_try:
        # Retry with all keys
        for i in range(len(GROQ_KEYS)):
            try:
                tools = get_fresh_tools()
                current_key = os.environ["GROQ_API_KEY"]

                llm = ChatGroq(model=model_name, temperature=0.1, api_key=current_key).bind_tools(tools)
                response = llm.invoke(state["messages"])
                return {"messages": [response]}

            except Exception as e:
                error_str = str(e)
                last_error = e

                if "429" in error_str or "Rate limit" in error_str:
                    print(f"⚠️ Key #{st.session_state.groq_idx + 1} hit limit. Rotating...")
                    rotate_groq_key()
                    continue
                elif "401" in error_str:
                    print(f"⚠️ Key #{st.session_state.groq_idx + 1} invalid. Rotating...")
                    rotate_groq_key()
                    continue
                else:
                    # Non-key error, try next model
                    print(f"❌ Model Error {model_name}: {error_str}")
                    break

    return {"messages": [
        AIMessage(content=f"❌ **System Exhausted:** All keys/models failed. Last Error: {str(last_error)}")],
            "final": True}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(get_fresh_tools()))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- 7. CHAT UI ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

if has_data:
    st.info(f"📊 **Analyst Mode:** Active")

history = load_history(current_sess)
current_messages = []

for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    avatar = theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
    if role == "user":
        current_messages.append(HumanMessage(content=msg["content"]))
    else:
        current_messages.append(AIMessage(content=msg["content"]))

if prompt := st.chat_input("Enter command..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    system_text = "You are Nexus."
    if has_data:
        # --- ANTI-LOOP SYSTEM PROMPT ---
        system_text += """
        [DATA MODE ACTIVE]
        1. Variable 'df' is loaded.
        2. CHECK COLUMNS FIRST: If unsure of column names, run `print(df.columns)` first.
        3. PLOTTING: Use `plt.figure()` -> `plt.plot()` -> NO `plt.show()`.
        4. STOPPING RULE: If the tool says "[SYSTEM MSG: Chart Successfully Rendered]", YOU MUST STOP. Do not run code again.
        5. If your code fails twice, stop and tell the user you cannot plot it.
        """
    else:
        system_text += " If no file, use 'tavily'."

    current_messages.append(SystemMessage(content=system_text))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        try:
            final_response = ""
            for event in app.stream({"messages": current_messages}, config={"recursion_limit": 50},
                                    stream_mode="values"):
                last_msg = event["messages"][-1]

                # --- LIVE DEBUGGING: Show tool output to User ---
                if isinstance(last_msg, ToolMessage):
                    status_box.write(f"⚙️ **Tool Output:** {last_msg.content[:200]}...")  # Show first 200 chars

                if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                    for t in last_msg.tool_calls:
                        status_box.write(f"⚙️ **Calling:** `{t['name']}`")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    st.markdown(final_response)

            if final_response:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_response)
            else:
                st.error("No response.")
        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")