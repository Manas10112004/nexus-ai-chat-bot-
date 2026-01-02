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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ System Halted: Missing API Keys.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# DEFINE MODEL TIERS
MODEL_FAST = "llama-3.1-8b-instant"  # Speed / Chat
MODEL_SMART = "llama-3.3-70b-versatile"  # Coding / Complex Tasks


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
            plt.clf()
            exec(code, self.scope)
            result = redirected_output.getvalue()

            if plt.get_fignums():
                st.pyplot(plt)
                plt.clf()
                return f"Output:\n{result}\n[SUCCESS: Chart Rendered. STOP.]"

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

# --- 4. SIDEBAR & ROUTING ---
current_theme = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme)
theme_data = THEMES.get(current_theme, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")

    # 🧠 SMART MODEL SELECTOR
    st.markdown("### 🧠 Intelligence")
    model_choice = st.radio(
        "Select Engine:",
        ["Auto (Smart)", "⚡ Fast (8B)", "🧙‍♂️ Genius (70B)"],
        index=0,
        help="Auto uses 70B for Data Analysis and 8B for Chat."
    )

    st.divider()

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


# --- 5. LOGIC & TOOLS ---
class PythonInput(BaseModel):
    code: str = Field(description="Python code to run. Use 'df'.")


def python_analysis_tool(code: str):
    return engine.run_python_analysis(code)


tavily = TavilySearchResults(max_results=2)
tools = [tavily]

has_data = "df" in engine.scope or "file_content" in engine.scope

if has_data:
    tools.append(StructuredTool.from_function(
        func=python_analysis_tool,
        name="python_analysis",
        description="Run Python analysis.",
        args_schema=PythonInput
    ))

# --- ROUTING LOGIC ---
if model_choice == "Auto (Smart)":
    # Use 70B if we have data (needs strict coding), otherwise 8B (speed)
    active_model = MODEL_SMART if has_data else MODEL_FAST
elif model_choice == "⚡ Fast (8B)":
    active_model = MODEL_FAST
else:
    active_model = MODEL_SMART

# Initialize LLM with selected model
llm = ChatGroq(model=active_model, temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


# --- 6. AGENT GRAPH ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- 7. CHAT UI ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

# Status Bar
st.caption(f"🚀 **Engine Active:** {active_model} | 📂 **Data Mode:** {'ON' if has_data else 'OFF'}")

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
        system_text += """
        [DATA MODE]
        1. Variable 'df' is loaded.
        2. Plotting: `plt.plot()`. NO `plt.show()`.
        """
    else:
        system_text += " If no file, use 'tavily'."

    current_messages.append(SystemMessage(content=system_text))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Thinking...", expanded=True)
        try:
            final_response = ""
            for event in app.stream({"messages": current_messages}, config={"recursion_limit": 50},
                                    stream_mode="values"):
                last_msg = event["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                    for t in last_msg.tool_calls:
                        status_box.write(f"⚙️ **{active_model} Using:** `{t['name']}`")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    st.markdown(final_response)

            if final_response:
                status_box.update(label="Done", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_response)
            else:
                st.error("No response.")
        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")