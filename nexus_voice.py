import streamlit as st
import os
from groq import Groq

client = None


def init_voice_client():
    global client
    # Fetches key dynamically from environment
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)


def transcribe_audio(audio_file):
    """
    Transcribes audio directly from memory (BytesIO).
    """
    if not client:
        init_voice_client()

    if not client or not audio_file:
        return None

    try:
        # 1. Rewind the virtual file to the start
        audio_file.seek(0)

        # 2. Send to Groq API
        transcription = client.audio.transcriptions.create(
            file=("input.wav", audio_file),  # Tuple tells API this is a WAV file
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )

        text = transcription.text.strip()

        # 3. Basic Filter (Only filter pure hallucination)
        if not text:
            return ""

        # Optional: Filter out "Thank you" hallucinations if text is VERY short
        if len(text) < 15 and text.lower().replace(".", "") in ["thank you", "you"]:
            return ""

        return text

    except Exception as e:
        # Return string with error so UI can show it
        return f"[API Error: {str(e)}]"