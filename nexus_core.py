import streamlit as st
import pandas as pd
import os
import time
import uuid
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
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# Database Imports
from nexus_db import init_db, save_message, load_history, clear_session, save_setting, load_setting, get_all_sessions
from themes import THEMES, inject_theme_css

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

# --- 2. CONFIGURATION ---
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ System Halted: Missing API Keys in Secrets.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
MODEL_NAME = "llama-3.3-70b-versatile"


# --- 3. ROBUST DATA ENGINE ---
class DataEngine:
    def __init__(self):
        self.df = None
        self.file_content = None
        self.repl = PythonREPL()

    def load_file(self, uploaded_file):
        try:
            name = uploaded_file.name
            # Tabular
            if name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif 'xls' in name:
                    self.df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    self.df = pd.read_json(uploaded_file)
                return f"✅ Data Loaded: {len(self.df)} rows, {len(self.df.columns)} columns."

            # Text
            elif name.endswith(('.txt', '.py', '.md', '.log', '.yaml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()
                return f"✅ Text Loaded: {len(self.file_content)} characters."

            else:
                return f"⚠️ Binary file '{name}' received. (Limited access)."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def run_python_analysis(self, code: str):
        # Inject variables so the AI can "see" them
        local_scope = {"df": self.df, "file_content": self.file_content, "pd": pd, "plt": plt, "sns": sns}
        try:
            return self.repl.run(code)
        except Exception as e:
            return f"Code Error: {str(e)}"


# --- 4. INITIALIZATION ---
def get_engine():
    if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
    return st.session_state.data_engine


def get_session():
    if "current_session_id" not in st.session_state: st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
    return st.session_state.current_session_id


engine = get_engine()
current_sess = get_session()
init_db()

# --- 5. SIDEBAR & FILE LOADER ---
current_theme = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme)
theme_data = THEMES.get(current_theme, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")
    uploaded_file = st.file_uploader("📂 Upload File", type=None)

    # LOAD FILE IMMEDIATELY
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

# --- 6. AGENT SETUP ---
tavily = TavilySearchResults(max_results=2)
tools = [tavily]


# DYNAMIC TOOL BINDING
def python_analysis_tool(code: str):
    return engine.run_python_analysis(code)


has_data = engine.df is not None or engine.file_content is not None

if has_data:
    tools.append(Tool(
        name="python_analysis",
        func=python_analysis_tool,
        description="EXECUTE PYTHON CODE. Use this to read 'df' (pandas dataframe) or 'file_content' (string)."
    ))

llm = ChatGroq(model=MODEL_NAME, temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


# GRAPH DEFINITION
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

# VISUAL DEBUGGER (So you know if Python sees the file)
if has_data:
    st.caption(f"🔵 **SYSTEM: Data Context Active** | Rows: {len(engine.df) if engine.df is not None else 'N/A'}")

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

    # --- CRITICAL: THE OVERRIDE PROMPT ---
    # This forces the LLM to admit it has the file.
    system_override = "You are Nexus."
    if has_data:
        system_override += """
        ⚠️ CRITICAL INSTRUCTION: A FILE IS LOADED.
        - You have a tool called 'python_analysis'.
        - You MUST use this tool to answer questions about the file.
        - The variable name is 'df' (if tabular) or 'file_content' (if text).
        - NEVER say 'I cannot access files'. You HAVE the tool. USE IT.
        """
    else:
        system_override += " If no file is loaded, use 'tavily' for web search."

    # Send System Message + User Prompt
    current_messages.append(SystemMessage(content=system_override))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Thinking...", expanded=True)
        final_response = ""

        try:
            for event in app.stream({"messages": current_messages}, stream_mode="values"):
                last_msg = event["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                    for t in last_msg.tool_calls:
                        status_box.write(f"⚙️ **Tool Active:** {t['name']}")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    st.markdown(final_response)

            if final_response:
                status_box.update(label="Done", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_response)
            else:
                st.error("No response from agent.")

        except Exception as e:
            st.error(f"Error: {e}")