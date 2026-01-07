import streamlit as st

# --- PROFESSIONAL THEME DEFINITION ---
THEMES = {
    "Nexus Enterprise": {
        "primary": "#3B82F6",  # Professional Royal Blue
        "background": "#0E1117",  # Deep Charcoal (Standard Streamlit Dark)
        "sidebar": "#161B22",  # Slightly lighter dark for sidebar
        "text": "#F3F4F6",  # Off-white for readability
        "user_avatar": "🧑‍💼",  # Professional User Icon
        "ai_avatar": "⚡",  # Minimalist AI Icon
        "font": "sans-serif"
    }
}


def inject_theme_css(theme_name):
    """Injects professional CSS styles into the app."""
    theme = THEMES.get(theme_name, THEMES["Nexus Enterprise"])

    css = f"""
    <style>
        /* 1. MAIN CONTAINER & FONT */
        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {theme['text']};
        }}

        /* 2. SIDEBAR STYLING */
        [data-testid="stSidebar"] {{
            background-color: {theme['sidebar']};
            border-right: 1px solid #30363D;
        }}

        [data-testid="stSidebar"] h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            margin-bottom: 1rem;
        }}

        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
            color: #8B949E; /* Muted text for sidebar labels */
            font-size: 0.85rem;
            font-weight: 500;
        }}

        /* 3. BUTTONS (SaaS Style) */
        .stButton button {{
            background-color: {theme['primary']};
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}

        .stButton button:hover {{
            background-color: #2563EB; /* Slightly darker blue on hover */
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transform: translateY(-1px);
        }}

        /* Secondary/Outline Buttons (if any) */
        div[data-testid="stForm"] .stButton button {{
             width: 100%;
        }}

        /* 4. CHAT MESSAGE STYLING */
        /* User Message Bubble */
        [data-testid="stChatMessage"]:nth-child(odd) {{
            background-color: transparent;
        }}

        /* AI Message Bubble (Subtle highlight) */
        [data-testid="stChatMessage"]:nth-child(even) {{
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 1rem;
        }}

        /* 5. METRIC CARDS & HEADERS */
        h1, h2, h3 {{
            color: #FFFFFF;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        /* 6. STATUS CONTAINERS (Success, Error, Info) */
        .stAlert {{
            border-radius: 6px;
            border: 1px solid #30363D;
        }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)