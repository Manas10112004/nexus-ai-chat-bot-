import streamlit as st
import os
from groq import Groq

# Reuse the existing key mechanism from your brain
client = None


def init_voice_client():
    global client
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)


def transcribe_audio(audio_bytes):
    if not client:
        init_voice_client()

    if not client or not audio_bytes:
        return None

    # Save temp file for Groq to read
    with open("temp_voice.wav", "wb") as f:
        f.write(audio_bytes)

    try:
        with open("temp_voice.wav", "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename("temp_voice.wav"), file.read()),
                # UPDATED MODEL NAME
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )
        return transcription.text
    except Exception as e:
        st.error(f"Voice Error: {e}")
        return None