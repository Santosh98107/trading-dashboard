import streamlit as st
from config import APP_USERNAME, APP_PASSWORD


def login_form():
    st.title("Trading Dashboard Login")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == APP_USERNAME and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    return False


def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
