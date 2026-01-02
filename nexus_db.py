import streamlit as st
from supabase import create_client, Client
import os

# --- CONNECT TO CLOUD ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def init_db():
    """Checks connection to Supabase."""
    if not supabase:
        st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY in Secrets.")
        st.stop()

def save_message(session_id, role, content):
    """Saves a message to the cloud."""
    data = {
        "session_id": session_id,
        "role": role,
        "content": content
    }
    supabase.table("messages").insert(data).execute()

def load_history(session_id):
    """Loads chat history for a specific session."""
    response = supabase.table("messages") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at", desc=False) \
        .execute()
    return response.data

def clear_session(session_id):
    """Deletes all messages for a specific session."""
    supabase.table("messages").delete().eq("session_id", session_id).execute()

def save_setting(key, value):
    """Upserts a setting."""
    data = {"key": key, "value": value}
    supabase.table("settings").upsert(data).execute()

def load_setting(key, default_value):
    """Loads a setting."""
    response = supabase.table("settings").select("value").eq("key", key).execute()
    if response.data:
        return response.data[0]['value']
    return default_value

def get_all_sessions():
    """Fetches unique session IDs sorted by latest activity."""
    # Note: Requires a 'messages' table.
    # Efficiency Note: Ideally, keep a separate 'sessions' table, but this works for simple apps.
    try:
        response = supabase.table("messages").select("session_id, created_at").order("created_at", desc=True).limit(500).execute()
        seen = set()
        sessions = []
        for row in response.data:
            s_id = row['session_id']
            if s_id not in seen:
                seen.add(s_id)
                sessions.append(s_id)
        return sessions
    except Exception:
        return []