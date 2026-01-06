import streamlit as st
import os
from groq import Groq

client = None


def init_voice_client():
    global client
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)


def transcribe_audio(audio_file):
    if not client:
        init_voice_client()

    if not client or not audio_file:
        return None

    # Save temp file (Standardizing input)
    try:
        # Reset file pointer to beginning just in case
        audio_file.seek(0)
        with open("temp_voice.wav", "wb") as f:
            f.write(audio_file.read())
    except Exception as e:
        st.error(f"File Error: {e}")
        return None

    try:
        with open("temp_voice.wav", "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename("temp_voice.wav"), file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )

        text = transcription.text.strip()

        # Simple Glitch Filter
        if not text or text.lower().startswith("thank you"):
            return ""

        return text

    except Exception as e:
        return f"[API Error: {str(e)}]"