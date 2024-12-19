import os

current_dir = os.path.dirname(__file__)


import streamlit as st
import json

# Load JSON data (simulating loading from a file)
chat_data = [{'role': 'user', 'content': 'A'}]

st.set_page_config(
    page_title="Reasoning",
    page_icon="graphics/Icon-BIDV.png" 
)

st.markdown("### Chat Messages")

with open('temp/sql_history.json', 'r') as file:
    chat_data = json.load(file)


# Render each message in markdown format
for message in chat_data:
    if message['role'] == 'user':
        with st.chat_message(name="user", avatar="graphics/user.jpg"):
            st.write(message['content'])
    if message['role'] == 'assistant':
        with st.chat_message(name="assistant", avatar="graphics/assistant.png"):
            st.write(message['content'])
