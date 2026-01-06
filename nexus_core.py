import streamlit as st
import uuid
import matplotlib.pyplot as plt
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from audio_recorder_streamlit import audio_recorder  # <--- VISUALIZER LIB

# --- CUSTOM MODULES ---
from nexus_db import init_db, save_message, load_history, clear_session, get_all_sessions, save_setting, load_setting
from themes import THEMES, inject_theme_css
from nexus_engine import DataEngine
from nexus_brain import build_agent_graph, get_key_status

# --- NEW MODULES ---
from nexus_security import check_password, logout
from nexus_report import generate_pdf
from nexus_voice import transcribe_audio

# --- UI CONFIG ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

# --- 1. SECURITY GATE ---
if not check_password():
    st.stop()

init_db()

# --- INITIALIZE STATE ---
if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
engine = st.session_state.data_engine

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
current_sess = st.session_state.current_session_id

# Track last voice input to prevent re-processing
if "last_voice_bytes" not in st.session_state:
    st.session_state.last_voice_bytes = None

# --- BUILD BRAIN ---
app = build_agent_graph(engine)

# --- SIDEBAR ---
current_theme = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme)
theme_data = THEMES.get(current_theme, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")
    st.caption(get_key_status())

    if st.button("🔒 Logout", use_container_width=True):
        logout()

    st.divider()

    # --- 2. VOICE MODE (VISUALIZER) ---
    st.markdown("### 🎙️ Voice Input")
    st.caption("Click to record. Click again to stop.")

    # Waveform Recorder
    voice_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x"
    )

    voice_text = ""
    # Check if we have NEW audio data
    if voice_bytes and voice_bytes != st.session_state.last_voice_bytes:
        # Show temporary status
        status_container = st.empty()
        status_container.info("Transcribing...")

        # Transcribe
        voice_text = transcribe_audio(voice_bytes)

        # Clear status
        status_container.empty()

        # Update State
        st.session_state.last_voice_bytes = voice_bytes

    st.divider()

    uploaded_file = st.file_uploader("📂 Upload File", type=None)
    if uploaded_file:
        status = engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

    if st.button("🧹 Clear Plots"):
        plt.clf()
        engine.latest_figure = None
        if os.path.exists("temp_chart.png"): os.remove("temp_chart.png")
        st.success("Plots cleared.")

    st.divider()

    # --- 3. SMART REPORTING ---
    st.markdown("### 📄 Reporting")
    if st.button("📥 Export PDF Report"):
        history = load_history(current_sess)
        pdf_file = generate_pdf(history, current_sess)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=pdf_file)

    st.divider()

    st.markdown("### 🕒 History")
    if st.button("➕ New Chat"):
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()

    for s in get_all_sessions()[:5]:
        if st.button(f"{s}", key=s):
            st.session_state.current_session_id = s
            st.rerun()

# --- CHAT INTERFACE ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

# Load History
history = load_history(current_sess)
for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]):
        st.markdown(msg["content"])

# --- INPUT HANDLING ---
user_input = st.chat_input("Enter command...")

prompt = None
# Prioritize Voice if fresh
if voice_text:
    prompt = voice_text
elif user_input:
    prompt = user_input

if prompt:
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    # Refresh Cheatsheet
    if engine.df is not None and not engine.column_str:
        engine.column_str = ", ".join(list(engine.df.columns))

    # --- SYSTEM PROMPT ---
    system_text = "You are Nexus, an advanced data analysis AI. You have access to a Python environment."
    has_data = "df" in engine.scope

    if has_data:
        system_text += f"""
        [DATA MODE ACTIVE]
        1. Variable 'df' is loaded.
        2. VALID COLUMNS: [{engine.column_str}]
        3. Use 'python_analysis' for all data queries.
        """
    else:
        # Enable Python even without a file (Sandbox Mode)
        system_text += """
        [SANDBOX MODE ACTIVE]
        1. No file loaded.
        2. You can still use 'python_analysis' to:
           - Generate synthetic data (e.g. "plot a sine wave", "simulate a random walk").
           - Perform math calculations.
        3. If asked for real-world facts/news, use 'tavily'.
        """

    # Memory Window
    recent_history = history[-6:]

    messages = [SystemMessage(content=system_text)] + \
               [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in
                recent_history] + \
               [HumanMessage(content=prompt)]

    # Run Agent
    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        try:
            final_resp = ""
            # 🟢 RECURSION LIMIT SET TO 10 TO PREVENT LOOPS
            for event in app.stream({"messages": messages}, config={"recursion_limit": 10}, stream_mode="values"):
                msg = event["messages"][-1]

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for t in msg.tool_calls:
                        status_box.write(f"⚙️ Calling: `{t['name']}`")

                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_resp = msg.content

            # 1. Render Chart
            if engine.latest_figure:
                st.pyplot(engine.latest_figure)
                chart_path = f"chart_{current_sess}.png"
                engine.latest_figure.savefig(chart_path)
                engine.latest_figure = None

            # 2. Render Text
            if final_resp:
                st.markdown(final_resp)
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_resp)
            else:
                status_box.update(label="Complete", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")