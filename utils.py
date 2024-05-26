"""
utils.py
"""
import os
import hmac

import streamlit as st
from openai import OpenAI

# Get secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Config
LAST_UPDATE_DATE = "2024-05-02"

# Initialise the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def render_custom_css() -> None:
    """
    Applies custom CSS
    """
    st.html("""
            <style>
                #MainMenu {visibility: hidden}
                #header {visibility: hidden}
                #footer {visibility: hidden}
                .block-container {
                    padding-top: 3rem;
                    padding-bottom: 2rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                    }
            </style>
            """)

def initialise_session_state():
    """
    Initialise session state variables
    """
    for session_state_var in ["file_uploaded", "read_terms"]:
        if session_state_var not in st.session_state:
            st.session_state[session_state_var] = False

def moderation_endpoint(text) -> bool:
    """
    Checks if the text is triggers the moderation endpoint

    Args:
    - text (str): The text to check

    Returns:
    - bool: True if the text is flagged
    """
    response = client.moderations.create(input=text)
    return response.results[0].flagged
