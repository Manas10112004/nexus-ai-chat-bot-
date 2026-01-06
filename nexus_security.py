import streamlit as st


def check_password():
    """Returns True if the user is logged in."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Login Form
    st.markdown("## 🔐 Nexus AI // Login")

    # Simple hardcoded credentials (in real production, use a DB)
    # You can also use st.secrets["ADMIN_USER"] and st.secrets["ADMIN_PASS"]
    REAL_USER = "admin"
    REAL_PASS = "nexus2026"

    with st.form("login_form"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Enter System")

        if submit:
            if user == REAL_USER and pwd == REAL_PASS:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("⛔ Access Denied")

    return False


def logout():
    st.session_state.authenticated = False
    st.rerun()