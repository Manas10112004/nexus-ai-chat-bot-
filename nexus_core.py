import streamlit as st
import uuid
import matplotlib.pyplot as plt
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- CUSTOM MODULES ---
from nexus_db import init_db, save_message, load_history, clear_session, get_all_sessions, save_setting, load_setting
from themes import THEMES, inject_theme_css
from nexus_engine import DataEngine
from nexus_brain import build_agent_graph, get_key_status

# --- SECURITY & REPORTING MODULES ---
from nexus_security import check_password, logout
from nexus_report import generate_pdf

# --- UI CONFIG ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

# --- 1. SECURITY GATE (Login Screen) ---
if not check_password():
    st.stop()

init_db()

# --- INITIALIZE STATE ---
if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
engine = st.session_state.data_engine

# --- MULTI-USER SESSION MANAGEMENT ---
# Grab the logged-in username
current_user = st.session_state.get("username", "guest")

# Prefix session ID with username to enforce data privacy
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"{current_user}-Session-{uuid.uuid4().hex[:4]}"

# Safety check: If user switched accounts, reset session
if not st.session_state.current_session_id.startswith(f"{current_user}-"):
    st.session_state.current_session_id = f"{current_user}-Session-{uuid.uuid4().hex[:4]}"

current_sess = st.session_state.current_session_id

# --- BUILD BRAIN ---
app = build_agent_graph(engine)

# --- SIDEBAR & THEME ---
# [FIX] Default to the professional theme
current_theme = load_setting("theme", "Nexus Enterprise")
inject_theme_css(current_theme)

# [FIX] Use the safe fallback if theme loading fails
theme_data = THEMES.get(current_theme, THEMES["Nexus Enterprise"])

with st.sidebar:
    st.title("⚡ NEXUS HQ")
    st.write(f"👤 **User:** {current_user}")
    st.caption(get_key_status())

    if st.button("🔒 Logout", use_container_width=True):
        logout()

    st.divider()

    # --- 2. DATA CENTER ---
    st.markdown("### 📂 Data Center")
    uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'xls', 'json'],
                                     help="Supported: CSV, Excel, JSON")

    if uploaded_file:
        status = engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

    if st.button("🧹 Clear Plots", use_container_width=True):
        plt.clf()
        engine.latest_figure = None
        if os.path.exists("temp_chart.png"): os.remove("temp_chart.png")
        st.success("Plots cleared.")

    st.divider()

    # --- 3. REPORTING ---
    st.markdown("### 📄 Reporting")
    if st.button("📥 Export PDF Report", use_container_width=True):
        with st.spinner("Compiling PDF..."):
            history = load_history(current_sess)
            pdf_file = generate_pdf(history, current_sess)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=pdf_file, use_container_width=True)

    st.divider()

    st.markdown("### 🕒 Session History")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New", use_container_width=True):
            st.session_state.current_session_id = f"{current_user}-Session-{uuid.uuid4().hex[:4]}"
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_session(current_sess)
            st.rerun()

    # List recent sessions (Filtered by User)
    st.caption("Recent Sessions:")
    my_sessions = [s for s in get_all_sessions() if s.startswith(f"{current_user}-")]

    for s in my_sessions[:5]:
        display_name = s.replace(f"{current_user}-", "")
        if st.button(f"📂 {display_name}", key=s, use_container_width=True):
            st.session_state.current_session_id = s
            st.rerun()

# --- CHAT INTERFACE ---
st.title("Nexus Intelligent Analytics")

# Load History
history = load_history(current_sess)
for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]):
        st.markdown(msg["content"])

# --- INPUT HANDLING ---
prompt = st.chat_input("Enter analysis command...")

if prompt:
    # 1. UI Echo
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    # 2. Refresh Context
    if engine.df is not None and not engine.column_str:
        engine.column_str = ", ".join(list(engine.df.columns))

    # 3. Construct System Prompt
    system_text = "You are Nexus, an advanced data analysis AI. You have access to a Python environment."
    has_data = "df" in engine.scope

    if has_data:
        system_text += f"""
        [DATA MODE ACTIVE]
        1. Variable 'df' is loaded.
        2. VALID COLUMNS: [{engine.column_str}]
        3. RULES:
           - Plan your step before writing code.
           - Use 'python_analysis' for all data queries.
           - When plotting, ALWAYS ensure the figure is created.
        """
    else:
        system_text += """
        [SANDBOX MODE ACTIVE]
        1. No file loaded.
        2. You can still use 'python_analysis' to:
           - Generate synthetic data.
           - Perform math calculations.
        3. If asked for real-world facts/news, use 'tavily'.
        """

    # 4. Context Window
    recent_history = history[-2:]

    messages = [SystemMessage(content=system_text)] + \
               [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in
                recent_history] + \
               [HumanMessage(content=prompt)]

    # 5. Run Agent
    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Thinking...", expanded=True)
        try:
            final_resp = ""
            # Stream the graph events
            for event in app.stream({"messages": messages}, config={"recursion_limit": 60}, stream_mode="values"):
                msg = event["messages"][-1]

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for t in msg.tool_calls:
                        status_box.write(f"⚙️ Action: `{t['name']}`")

                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_resp = msg.content

            # A. Render Chart (if generated)
            if engine.latest_figure:
                st.pyplot(engine.latest_figure)
                chart_path = f"chart_{current_sess}.png"
                engine.latest_figure.savefig(chart_path)
                engine.latest_figure = None

            # B. Render Text Response
            if final_resp:
                st.markdown(final_resp)
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_resp)
            else:
                status_box.update(label="Task Completed", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")