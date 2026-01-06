import streamlit as st
import uuid
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from groq import Groq

# --- CUSTOM MODULES ---
from nexus_db import init_db, save_message, load_history, clear_session, get_all_sessions, save_setting, load_setting
from themes import THEMES, inject_theme_css
from nexus_engine import DataEngine
from nexus_brain import build_agent_graph, get_key_status

# --- UI CONFIG ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")
init_db()

# --- INITIALIZE STATE ---
if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
engine = st.session_state.data_engine

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
current_sess = st.session_state.current_session_id

if "last_audio_buffer" not in st.session_state: st.session_state.last_audio_buffer = None

# --- BUILD BRAIN ---
app = build_agent_graph(engine)

# --- SIDEBAR ---
current_theme = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme)
theme_data = THEMES.get(current_theme, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")
    st.caption(get_key_status())

    # --- 🎙️ VOICE INPUT ---
    st.divider()
    st.markdown("### 🎙️ Voice Mode")
    audio_input = st.audio_input("Record Voice Command")

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
        st.success("Plots cleared.")

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

# --- CHAT INTERFACE ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

history = load_history(current_sess)
for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]):
        st.markdown(msg["content"])

# --- INPUT LOGIC (FIXED) ---
# 1. Always render the text box so it never disappears
text_input = st.chat_input("Enter command...")

# 2. Determine final prompt (Voice > Text)
prompt = None
voice_detected = False

# Check Voice (Only if new)
if audio_input is not None:
    current_audio_signature = audio_input.getvalue()
    if current_audio_signature != st.session_state.last_audio_buffer:
        st.session_state.last_audio_buffer = current_audio_signature
        try:
            with st.spinner("🎧 Transcribing..."):
                client = Groq(api_key=st.secrets["GROQ_API_KEYS"].split(",")[0])
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", audio_input, "audio/wav"),
                    model="whisper-large-v3-turbo",
                    response_format="json"
                )
                prompt = transcription.text
                voice_detected = True
        except Exception as e:
            st.error(f"Transcription Failed: {e}")

# If no fresh voice command, check if user typed something
if not prompt and text_input:
    prompt = text_input

# --- EXECUTION ---
if prompt:
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        if voice_detected:
            st.markdown(f"🎙️ *{prompt}*")
        else:
            st.markdown(prompt)

    save_message(current_sess, "user", prompt)

    if engine.df is not None and not engine.column_str:
        engine.column_str = ", ".join(list(engine.df.columns))

    system_text = "You are Nexus."
    has_data = "df" in engine.scope
    if has_data:
        system_text += f"""
        [DATA MODE ACTIVE]
        1. Variable 'df' is loaded.
        2. VALID COLUMNS: [{engine.column_str}]

        3. ⚠️ MANDATORY TOOL USAGE ⚠️
           - If user asks for "ANOMALIES" or "OUTLIERS" -> YOU MUST USE: `insights.check_anomalies(df, 'col')`.
           - If user asks for "FORECAST" or "PREDICT" -> YOU MUST USE: `insights.forecast_series(df, 'date', 'val', 30)`.
           - If user asks for "DRIVERS" or "CORRELATION" -> YOU MUST USE: `insights.get_correlation_drivers(df, 'target')`.

        4. FOR SIMPLE QUESTIONS:
           - Use `print()` for numbers.
           - Use `plt.plot()` for charts.
        """
    else:
        system_text += " If no file, use 'tavily'."

    recent_history = history[-6:]

    messages = [SystemMessage(content=system_text)] + \
               [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in
                recent_history] + \
               [HumanMessage(content=prompt)]

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        try:
            final_resp = ""
            for event in app.stream({"messages": messages}, config={"recursion_limit": 50}, stream_mode="values"):
                msg = event["messages"][-1]

                if isinstance(msg, ToolMessage):
                    status_box.write(f"⚙️ Output: {msg.content[:200]}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for t in msg.tool_calls:
                        status_box.write(f"⚙️ Calling: `{t['name']}`")
                if isinstance(msg, AIMessage) and msg.content:
                    final_resp = msg.content
                    st.markdown(final_resp)

            if engine.latest_figure:
                st.pyplot(engine.latest_figure)
                engine.latest_figure = None

            if final_resp:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_resp)
            else:
                st.error("No response.")

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")