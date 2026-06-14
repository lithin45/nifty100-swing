"""Password gate for the dashboard.

Simple secrets-based protection (``st.secrets["app_password"]`` or the
``APP_PASSWORD`` env var). For multi-user logins you can swap this for
``streamlit-authenticator`` — the rest of the app only calls
:func:`require_password`.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st


def _configured_password() -> str:
    try:
        if "app_password" in st.secrets:
            return str(st.secrets["app_password"])
    except Exception:
        pass
    return os.getenv("APP_PASSWORD", "")


def require_password() -> bool:
    """Render a login form and return True once authenticated.

    If no password is configured, the dashboard is left open (with a warning) so
    local development is frictionless.
    """
    correct = _configured_password()
    if not correct:
        st.warning("No `app_password` set in secrets — dashboard is unprotected. "
                   "Add one before deploying publicly.")
        return True

    if st.session_state.get("auth_ok"):
        return True

    st.markdown("### 🔒 Protected dashboard")
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(pw, correct):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return bool(st.session_state.get("auth_ok"))
