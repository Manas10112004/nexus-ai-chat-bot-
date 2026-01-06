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
    return f"Keys: Groq({st.session_state.groq_idx + 1}) | Tavily({st.session_state.tavily_idx + 1})"


# --- AGENT SETUP ---
class PythonInput(BaseModel):
    code: str = Field(description="Python code. Use 'df'.")


def get_tools(data_engine):
    update_env_vars()
    tools = [TavilySearchResults(max_results=2)]

    # Only enable Python tool if data exists
    has_data = "df" in data_engine.scope or "file_content" in data_engine.scope
    if has_data:
        def python_wrapper(code: str):
            return data_engine.run_python_analysis(code)

        tools.append(StructuredTool.from_function(
            func=python_wrapper,
            name="python_analysis",
            description="Run Python code.",
            args_schema=PythonInput
        ))
    return tools


# --- AGENT GRAPH ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def build_agent_graph(data_engine):
    init_keys()

    def agent_node(state):
        # --- SHORT-CIRCUIT LOGIC (FIXED) ---
        last_msg = state["messages"][-1]

        if isinstance(last_msg, ToolMessage):
            content = last_msg.content

            # Helper to extract text
            def extract_text(raw):
                clean = raw.replace("[CHART GENERATED]", "").replace("[ANALYSIS COMPLETE]", "").replace("Output:\n",
                                                                                                        "").strip()
                return clean

            # Case A: Chart Created
            if "[CHART GENERATED]" in content:
                explanation = extract_text(content)
                if not explanation: explanation = "Chart generated (No text description provided)."
                return {"messages": [AIMessage(content=f"{explanation}\n\n*(See the chart plotted above)*")]}

            # Case B: Text Analysis
            if "[ANALYSIS COMPLETE]" in content:
                clean_answer = extract_text(content)
                if not clean_answer: clean_answer = "Analysis finished, but no text output was captured."
                return {"messages": [AIMessage(content=f"**Analysis Results:**\n\n{clean_answer}")]}

        # --- STANDARD LLM EXECUTION ---
        has_data = "df" in data_engine.scope
        primary_model = MODEL_SMART if has_data else MODEL_FAST
        models = [primary_model, MODEL_FAST] if primary_model != MODEL_FAST else [MODEL_FAST]

        last_error = None
        for model in models:
            for i in range(len(GROQ_KEYS)):
                try:
                    tools = get_tools(data_engine)
                    key = os.environ["GROQ_API_KEY"]

                    llm = ChatGroq(model=model, temperature=0.1, api_key=key).bind_tools(tools,
                                                                                         parallel_tool_calls=False)

                    return {"messages": [llm.invoke(state["messages"])]}
                except Exception as e:
                    last_error = e
                    if "429" in str(e) or "Rate limit" in str(e):
                        rotate_groq_key();
                        continue
                    elif "400" in str(e):
                        continue
                    break

        return {"messages": [AIMessage(content=f"❌ System Exhausted. Error: {str(last_error)}")], "final": True}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(get_tools(data_engine)))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()