import streamlit as st
import pandas as pd
import os
import time
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# Database Imports
from nexus_db import init_db, save_message, load_history, clear_session, save_setting, load_setting, get_all_sessions
from themes import THEMES, inject_theme_css

# --- 1. PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

# --- 2. CONFIGURATION ---
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ System Halted: Missing API Keys. Please add them to Streamlit Secrets.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

MODEL_NAME = "llama-3.3-70b-versatile"


# --- 3. DATA ENGINE CLASS ---
class DataEngine:
    """Handles file processing for ALL file types."""

    def __init__(self):
        self.df = None
        self.file_content = None
        self.file_type = None
        self.repl = PythonREPL()

    def load_file(self, uploaded_file):
        try:
            name = uploaded_file.name
            self.file_type = name.split('.')[-1].lower()

            # Tabular Data
            if name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif 'xls' in name:
                    self.df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    self.df = pd.read_json(uploaded_file)
                return f"✅ Dataset loaded. Shape: {self.df.shape}. Available as variable 'df'."

            # Text Data
            elif name.endswith(('.txt', '.py', '.md', '.log', '.toml', '.yml', '.yaml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()
                return f"✅ Text file read ({len(self.file_content)} chars). Available as variable 'file_content'."

            # Other Files
            else:
                return f"⚠️ File '{name}' received. (Binary/Complex files are currently read-only placeholders)."

        except Exception as e:
            return f"❌ Error loading file: {str(e)}"

    def run_python_analysis(self, code: str):
        try:
            local_scope = {
                "df": self.df,
                "file_content": self.file_content,
                "pd": pd,
                "plt": plt,
                "sns": sns,
                "st": st
            }
            return self.repl.run(code)
        except Exception as e:
            return f"Execution Error: {str(e)}"


# --- 4. INITIALIZATION (CRITICAL: MUST BE BEFORE SIDEBAR) ---
if "data_engine" not in st.session_state:
    st.session_state.data_engine = DataEngine()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"

init_db()

# --- 5. THEME & UI SETUP ---
current_theme_setting = load_setting("theme", "🌿 Eywa (Avatar)")
if current_theme_setting not in THEMES:
    current_theme_setting = "🌿 Eywa (Avatar)"
inject_theme_css(current_theme_setting)
theme_data = THEMES[current_theme_setting]

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ NEXUS HQ")

    # File Uploader
    uploaded_file = st.file_uploader("📂 Upload Data / Files", type=None)
    if uploaded_file:
        # data_engine is GUARANTEED to exist now because Step 4 ran first
        status = st.session_state.data_engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

    st.divider()

    # Chat Controls
    col1, col2 = st.columns(2)
    if col1.button("➕ New Chat"):
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()
    if col2.button("🗑️ Clear"):
        clear_session(st.session_state.current_session_id)
        st.rerun()

    # History List
    st.markdown("### 🕒 History")
    all_sessions = get_all_sessions()

    for sess in all_sessions[:8]:
        if sess == st.session_state.current_session_id:
            st.markdown(f"**🔹 {sess}**")
        else:
            if st.button(f"{sess}", key=f"btn_{sess}"):
                st.session_state.current_session_id = sess
                st.rerun()

    st.divider()

    selected_theme = st.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(current_theme_setting))
    if selected_theme != current_theme_setting:
        save_setting("theme", selected_theme)
        st.rerun()

    web_search_on = st.toggle("🌐 Web Search", value=True)

# --- 7. AI GRAPH & TOOLS ---
tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]


def python_analysis_tool(code: str):
    return st.session_state.data_engine.run_python_analysis(code)


# Only enable Python tool if data is actually loaded
if st.session_state.data_engine.df is not None or st.session_state.data_engine.file_content is not None:
    tools.append(Tool(
        name="python_analysis",
        func=python_analysis_tool,
        description="Run Python code. Variables: 'df' (pandas), 'file_content' (str). Use print() to see output."
    ))

llm = ChatGroq(model=MODEL_NAME, temperature=0.1)

if web_search_on or len(tools) > 1:
    llm_with_tools = llm.bind_tools(tools)
else:
    llm_with_tools = llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)

if web_search_on or len(tools) > 1:
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
else:
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

app = workflow.compile()

# --- 8. CHAT INTERFACE ---
st.title(f"NEXUS // {selected_theme.split(' ')[1].upper()}")

history = load_history(st.session_state.current_session_id)
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

if prompt := st.chat_input("Enter command or query..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(st.session_state.current_session_id, "user", prompt)

    system_text = """You are Nexus.
    1. If the user asks about the UPLOADED FILE, use 'python_analysis' tool.
       - 'df' is the dataframe. 'file_content' is the text string.
    2. If the user asks for WEB INFO, use 'tavily' tool.
    3. Always be concise and professional.
    """

    current_messages.append(SystemMessage(content=system_text))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        message_placeholder = st.empty()
        final_response = ""

        try:
            for event in app.stream({"messages": current_messages}, stream_mode="values"):
                last_msg = event["messages"][-1]

                if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                    for t in last_msg.tool_calls:
                        if t['name'] == 'python_analysis':
                            status_box.write(f"📊 **Analyzing Data:**\n```python\n{t['args']}\n```")
                        else:
                            status_box.write(f"🔍 **Searching:** {t['args']}")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    message_placeholder.markdown(final_response)

            if final_response:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(st.session_state.current_session_id, "assistant", final_response)
            else:
                status_box.update(label="Failed", state="error")
                st.error("No response generated.")

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {str(e)}")