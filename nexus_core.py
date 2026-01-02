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
from langchain_core.tools import Tool

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
MODEL_NAME = "llama-3.3-70b-versatile"


# --- 2. IMPROVED DATA ENGINE (THE FIX) ---
class DataEngine:
    def __init__(self):
        # We keep a persistent dictionary for variables
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

            # 1. LOAD DATAFRAME
            if name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif 'xls' in name:
                    self.df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    self.df = pd.read_json(uploaded_file)

                # INJECT INTO SCOPE
                self.scope["df"] = self.df
                return f"✅ Data Loaded: {len(self.df)} rows. (Accessible as 'df')"

            # 2. LOAD TEXT
            elif name.endswith(('.txt', '.py', '.md', '.log', '.yaml', '.xml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()

                # INJECT INTO SCOPE
                self.scope["file_content"] = self.file_content
                return f"✅ Text Loaded: {len(self.file_content)} chars. (Accessible as 'file_content')"

            else:
                return f"⚠️ Binary file '{name}' (Limited Access)."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def run_python_analysis(self, code: str):
        # CAPTURE OUTPUT
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            # EXECUTE CODE WITH PERSISTENT SCOPE
            exec(code, self.scope)
            result = redirected_output.getvalue()
            return f"Output:\n{result}" if result else "Code executed successfully (No output)."
        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout


# --- 3. INIT STATE ---
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
    uploaded_file = st.file_uploader("📂 Upload File", type=None)

    # Load File Immediately
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

# --- 5. DEFINE TOOLS ---
tavily = TavilySearchResults(max_results=2)
tools = [tavily]

has_data = "df" in engine.scope or "file_content" in engine.scope


def python_analysis_tool(code: str):
    return engine.run_python_analysis(code)


if has_data:
    tools.append(Tool(
        name="python_analysis",
        func=python_analysis_tool,
        description="EXECUTE PYTHON. You have variables: 'df' (pandas) or 'file_content' (str). PRINT outputs."
    ))

# --- 6. AGENT SETUP ---
llm = ChatGroq(model=MODEL_NAME, temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


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

if has_data:
    st.info(f"📁 **System Ready:** Data Loaded in variable 'df'. Tools Active: {len(tools)}")

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

    # --- SYSTEM OVERRIDE PROMPT ---
    system_text = "You are Nexus."
    if has_data:
        system_text += """
        [DATA MODE ACTIVE]
        - A dataframe is loaded as variable `df`.
        - You MUST use the `python_analysis` tool to inspect it.
        - To see data: Run `print(df.head())`
        - To get columns: Run `print(df.columns)`
        - DO NOT guess. Run code to see the data.
        """
    else:
        system_text += " If no file is loaded, use 'tavily' for web search."

    current_messages.append(SystemMessage(content=system_text))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Thinking...", expanded=True)
        try:
            final_response = ""
            for event in app.stream({"messages": current_messages}, stream_mode="values"):
                last_msg = event["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                    for t in last_msg.tool_calls:
                        status_box.write(f"⚙️ **Running Tool:** `{t['name']}`")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    st.markdown(final_response)

            if final_response:
                status_box.update(label="Done", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_response)
            else:
                st.error("No response.")
        except Exception as e:
            st.error(f"Error: {e}")