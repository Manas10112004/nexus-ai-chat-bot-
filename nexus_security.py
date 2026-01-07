import streamlit as st
import bcrypt
from nexus_db import get_supabase_client


# --- AUTHENTICATION FUNCTIONS ---

def hash_password(password):
    """Converts a plain password into a secure hash."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password, hashed_password):
    """Checks if the password matches the hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_user(username, password):
    """Registers a new user in Supabase."""
    supabase = get_supabase_client()

    # 1. Check if user already exists
    existing = supabase.table("users").select("username").eq("username", username).execute()
    if existing.data:
        return False, "Username already taken."

    # 2. Create new user
    hashed = hash_password(password)

    # [UPDATED] Include default plan_type
    data = {
        "username": username,
        "password_hash": hashed,
        "plan_type": "free"
    }

    try:
        supabase.table("users").insert(data).execute()
        return True, "Account created! You can now log in."
    except Exception as e:
        return False, f"Error: {str(e)}"


def login_user(username, password):
    """Verifies credentials."""
    supabase = get_supabase_client()

    # Fetch user
    response = supabase.table("users").select("*").eq("username", username).execute()

    if not response.data:
        return False

    user_data = response.data[0]
    stored_hash = user_data["password_hash"]

    if verify_password(password, stored_hash):
        return True
    return False


# --- UI COMPONENTS ---

def login_form():
    """Renders the Login/Signup UI."""
    st.markdown("## 🔐 Nexus AI Login")

    tab1, tab2 = st.tabs(["Log In", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")

            if submitted:
                if login_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success("Welcome back!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("Choose a Username")
            new_pass = st.text_input("Choose a Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")

            if submitted:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match!")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, msg = create_user(new_user, new_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)


def check_password():
    """Main gatekeeper function."""
    if st.session_state.get("authenticated", False):
        return True

    login_form()
    return False


def logout():
    """Clears session and logs out."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()