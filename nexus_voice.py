import streamlit as st
import os
from groq import Groq

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

    # Save temp file
    with open("temp_voice.wav", "wb") as f:
        f.write(audio_bytes)

    try:
        with open("temp_voice.wav", "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename("temp_voice.wav"), file.read()),
                # ✅ USE STABLE TURBO MODEL
                model="whisper-large-v3-turbo",
                # ✅ REMOVED THE "PROMPT" THAT CAUSED LOOPS
                response_format="json",
                language="en",
                temperature=0.0
            )

        text = transcription.text.strip()

        # ✅ MINIMALIST FILTER (Only blocks known glitches)
        # This ensures we don't accidentally block real words
        hallucinations = [
            "Thank you.", "Thank you", "Thanks.", "You", "MBC",
            "Amara.org", "Subtitles by", "Copyright", "Watching",
            "Thank you for watching"
        ]

        # Block only if it is EXACTLY a hallucination
        if text in hallucinations or text.lower().startswith("thank you for"):
            return ""

        return text

    except Exception as e:
        return None