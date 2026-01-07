import streamlit as st
import os
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# We prioritize the 70b model for logic, but fallback to 8b if needed
MODEL_SMART = "llama-3.3-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"

# Load Keys
raw_groq = st.secrets.get("GROQ_API_KEYS", "")
raw_tavily = st.secrets.get("TAVILY_API_KEYS", "")
GROQ_KEYS = [k.strip() for k in raw_groq.split(",") if k.strip()]
TAVILY_KEYS = [k.strip() for k in raw_tavily.split(",") if k.strip()]

if not GROQ_KEYS or not TAVILY_KEYS:
    st.error("⚠️ System Halted: Missing API Keys in Secrets.")
    st.stop()


# --- KEY MANAGEMENT ---
def init_keys():
    if "groq_idx" not in st.session_state: st.session_state.groq_idx = 0
    if "tavily_idx" not in st.session_state: st.session_state.tavily_idx = 0
    update_env_vars()


def update_env_vars():
    g_key = GROQ_KEYS[st.session_state.groq_idx % len(GROQ_KEYS)]
    t_key = TAVILY_KEYS[st.session_state.tavily_idx % len(TAVILY_KEYS)]
    os.environ["GROQ_API_KEY"] = g_key
    os.environ["TAVILY_API_KEY"] = t_key


def rotate_groq_key():
    st.session_state.groq_idx = (st.session_state.groq_idx + 1) % len(GROQ_KEYS)
    update_env_vars()


def get_key_status():
    return f"Active Key: Groq-{st.session_state.groq_idx + 1}"


# --- AGENT SETUP ---
class PythonInput(BaseModel):
    code: str = Field(description="Python code to execute. Always print output.")


def get_tools(data_engine):
    update_env_vars()

    # Tool 1: Web Search
    search = TavilySearchResults(max_results=2)

    # Tool 2: Python Engine
    def python_wrapper(code: str):
        return data_engine.run_python_analysis(code)

    python_tool = StructuredTool.from_function(
        func=python_wrapper,
        name="python_analysis",
        description="Executes Python code. Access 'df' (pandas DataFrame). Use plt.show() for plots.",
        args_schema=PythonInput
    )

    return [search, python_tool]


# --- AGENT GRAPH ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def build_agent_graph(data_engine):
    init_keys()

    def agent_node(state):
        # Try SMART model first, then FAST model
        models_to_try = [MODEL_SMART, MODEL_FAST]

        last_error = None

        for model_name in models_to_try:
            try:
                tools = get_tools(data_engine)
                key = os.environ["GROQ_API_KEY"]

                # CRITICAL: parallel_tool_calls=False prevents the "Double Code" bug
                llm = ChatGroq(
                    model=model_name,
                    temperature=0.0,
                    api_key=key
                ).bind_tools(tools, parallel_tool_calls=False)

                response = llm.invoke(state["messages"])
                return {"messages": [response]}

            except Exception as e:
                # If it's a Rate Limit (429), rotate key and retry SAME model
                if "429" in str(e) or "Rate limit" in str(e):
                    rotate_groq_key()
                    continue

                    # If it's a Model Overload (503/500), try NEXT model
                last_error = e
                continue

        # If all fail
        return {"messages": [AIMessage(content=f"❌ System Busy. Error: {str(last_error)}")]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(get_tools(data_engine)))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()