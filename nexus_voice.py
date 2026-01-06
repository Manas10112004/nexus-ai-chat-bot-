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
                model="whisper-large-v3-turbo",

                # 🛑 FIX 1: Change Prompt to a generic example, not an instruction.
                # This prevents the AI from parroting "SQL. Plot." back to you.
                prompt="The profit margin for 2024 was ten percent.",

                response_format="json",
                language="en",
                temperature=0.0
            )

        text = transcription.text.strip()

        # 🛑 FIX 2: Add known "Prompt Leaks" to the block list
        hallucinations = [
            "Thank you.", "Thank you", "Thanks.", "You",
            "MBC", "Amara.org", "Subtitles by", "Copyright",
            "Thank you for watching", "I'm going to go to sleep",
            "Bye", "Watching", "SQL. Plot. Calculate. No conversation.",
            "User command for data analysis."
        ]

        # Filter logic: Block if text is in list OR matches the prompt style
        if any(h.lower() in text.lower() for h in hallucinations) or len(text) < 2:
            return ""

        return text

    except Exception as e:
        # Hide minor errors to prevent UI clutter
        print(f"Voice Error: {e}")
        return None