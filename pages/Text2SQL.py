import os
import openai
import streamlit as st

import sys 
# sys.path.append("..")

from agent import Chatbot, Text2SQL, ChatbotSematic
from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    GEMINI_FAST_CONFIG_V2,
    GPT4O_MINI_CONFIG,
    GPT4O_CONFIG,
    GEMINI_EXP_CONFIG,
    INBETWEEN_CHAT_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_THINKING_GEMINI_CONFIG,
    TEXT2SQL_4O_CONFIG,
    TEXT2SQL_QWEN25_CODER_3B_SFT_CONFIG,
    TEXT2SQL_QWEN25_CODER_1B_KTO_CONFIG,
    TEXT2SQL_4O_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY_EXTEND,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI_EXTEND
)

from ETL.dbmanager import get_semantic_layer, BaseRerannk
import json
import torch
from initialize import initialize_text2sql

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


st.set_page_config(
    page_title="Chatbot",
    page_icon="graphics/Icon-BIDV.png" 
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# =======  Versioning ======== #
st.session_state.version = "v3.2"
st.session_state.chat_version = "v2"
st.session_state.rotate_api = True

if "history" not in st.session_state:
    st.session_state.history = []
# ============================ #

@st.cache_resource
def initialize(text2sql_model = 'gemini-2.0-flash'):
    
    prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI_EXTEND
    text2sql_config = TEXT2SQL_FAST_GEMINI_CONFIG

    if 'qwen2.5-3b-sft' in text2sql_model:
        text2sql_config = TEXT2SQL_QWEN25_CODER_3B_SFT_CONFIG
    elif 'qwen2.5-1.5b-kto' in text2sql_model:
        text2sql_config = TEXT2SQL_QWEN25_CODER_1B_KTO_CONFIG
    elif 'gpt-4o-mini' in text2sql_model:
        text2sql_config = TEXT2SQL_FAST_OPENAI_CONFIG
    elif 'gpt-4o' in text2sql_model:
        text2sql_config = TEXT2SQL_4O_CONFIG
    elif 'gemini-2.0-flash-thinking-exp-01-21' in text2sql_model:
        text2sql_config = TEXT2SQL_THINKING_GEMINI_CONFIG
    
    chatbot = initialize_text2sql(text2sql_config, prompt_config, version = st.session_state.version, message=True, rotate_key = st.session_state.rotate_api)

    st.session_state.history = []
    return chatbot



def chat(user_name):
    user_name = str(user_name)

    if "enhance" not in st.session_state:
        st.session_state.enhance = None

    if "text2sql_model" not in st.session_state:
        st.session_state.text2sql_model = 'gemini-2.0-flash'
    
    chatbot = initialize(text2sql_model=str(st.session_state.text2sql_model))
    st.session_state.chatbot = chatbot

    enhance = st.selectbox(
        "Enhance:",
        ['refinement', 'correction', None],
        index=['refinement', 'correction', None].index(st.session_state.enhance)
    )
    st.session_state.enhance = enhance
    
    text2sql_model = st.selectbox(
        "Text2SQL Model:",
        ['gemini-2.0-flash', 'qwen2.5-3b-sft', 'gpt-4o-mini', 'gpt-4o', 'gemini-2.0-flash-thinking-exp-01-21'],
        index=['gemini-2.0-flash', 'qwen2.5-3b-sft', 'gpt-4o-mini' , 'gpt-4o', 'gemini-2.0-flash-thinking-exp-01-21'].index(st.session_state.text2sql_model)
    )

    if text2sql_model != st.session_state.text2sql_model:
        st.session_state.text2sql_model = text2sql_model
        chatbot = initialize(text2sql_model=str(st.session_state.text2sql_model))
        st.session_state.chatbot = chatbot


    

    with st.container():     
        if st.button("Clear Chat"):
            st.session_state.chatbot.reset()
            st.session_state.history = []

    with st.chat_message( name="system"):
        st.markdown("© 2025 Nguyen Quang Hung. All rights reserved.")

    for message in st.session_state.history:
        if message['role'] == 'user':
            with st.chat_message(name="user", avatar="graphics/user.jpg"):
                st.write(message['content'])
        if message['role'] == 'assistant':
            with st.chat_message(name="assistant", avatar="graphics/assistant.png"):
                st.write(message['content'])
                
    input_text = st.chat_input("Chat with your bot here")   

    if input_text:
        with st.chat_message("user", avatar='graphics/user.jpg'):
            st.markdown(input_text)

        st.session_state.history.append({'role': 'user', 'content': input_text})

        
        assistant_message = st.chat_message("assistant", avatar='graphics/assistant.png').empty()   
        
        streamed_text = ""
        for chunk in st.session_state.chatbot.stream(input_text, version=st.session_state.chat_version, enhance = st.session_state.enhance):
            if isinstance(chunk, str):
                streamed_text += chunk
                assistant_message.write(streamed_text)
        st.session_state.history.append({'role': 'assistant', 'content': streamed_text})
          
        

     

users = {
    'admin': 'admin',
    'user': '12345678',
    'hanni': '12345678',
    'hung20gg': '12345678',
    'dpg': '12345678',
    'vybeo': '12345678',
    'quoc': '12345678',
    'tpa': '12345678',
    'ntp': '12345678',
    'phong': '12345678',
    'ngothao': '12345678',
    'ngan': '12345678',
    'synthetic': '12345678',
}     
     

def login():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if users.get(username, r'!!@@&&$$%%.') == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f'Welcome back {username}')
        else:
            st.error('Invalid username or password')
            
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success('You have been logged out')
    
def main():
    
    if st.session_state.logged_in:
        st.write(f'Logged in as {st.session_state.username}')
        chat(st.session_state.username)
    else:
        st.title('Welcome!!!')
        st.write('Press Login button 2 times to login')
        login()
        
main()

# chat('test')