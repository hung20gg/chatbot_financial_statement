# Copy this file to other directory and run it to see the chat interface
# There is some conflict between steamlit and chroma


import streamlit as st
import json

# Load JSON data (simulating loading from a file)
chat_data = [{'role': 'user', 'content': 'A'}]

# Streamlit app
st.title("Chat Interface")

st.markdown("### Chat Messages")

with open('history.json', 'r') as file:
    chat_data = json.load(file)


# Render each message in markdown format
for message in chat_data:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
