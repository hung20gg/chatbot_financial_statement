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
    GPT4O_MINI_CONFIG,
    GPT4O_CONFIG,
    GEMINI_EXP_CONFIG,
    INBETWEEN_CHAT_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_QWEN25_CODER_3B_SFT_CONFIG,
    TEXT2SQL_4O_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI
)

from ETL.dbmanager.setup import (
    DBConfig,
    BGE_VERTICAL_BASE_CONFIG,
    BGE_VERTICAL_UNIVERSAL_CONFIG,
    OPENAI_VERTICAL_UNIVERSAL_CONFIG,
    TEI_VERTICAL_UNIVERSAL_CONFIG,
    setup_db
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

@st.cache_resource
def initialize(user_name, chat_model = 'gemini-2.0-flash', text2sql_model = 'gemini-2.0-flash'):
    
    prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI
    text2sql_config = TEXT2SQL_FAST_GEMINI_CONFIG
    chat_config = GEMINI_FAST_CONFIG

    if 'gemini-2.0-flash' in chat_model:
        chat_config = GEMINI_FAST_CONFIG
    if 'gpt-4o-mini' in chat_model:
        chat_config = GPT4O_MINI_CONFIG
    elif 'gpt-4o' in chat_model:
        chat_config = GPT4O_CONFIG

    if 'qwen2.5-3b' in text2sql_model:
        text2sql_config = TEXT2SQL_QWEN25_CODER_3B_SFT_CONFIG
    elif 'gpt-4o-mini' in text2sql_model:
        text2sql_config = TEXT2SQL_FAST_OPENAI_CONFIG
    
    text2sql = initialize_text2sql(text2sql_config, prompt_config)
    
    message_saver = get_semantic_layer()
    
    chatbot = ChatbotSematic(config = chat_config, text2sql = text2sql, message_saver = message_saver)
    logging.info('Finish setup chatbot')
    
    chatbot.create_new_chat(user_id=user_name)
    
    
    return chatbot



def chat(user_name):
    user_name = str(user_name)

    st.session_state.chat_model = 'gemini-2.0-flash'
    st.session_state.text2sql_model = 'gemini-2.0-flash'
    
    chatbot = initialize(user_name, text2sql_model=str(st.session_state.text2sql_model), chat_model=str(st.session_state.chat_model))
    st.session_state.chatbot = chatbot

    chat_model = st.selectbox(
        "Chat Model:",
        ['gemini-2.0-flash', 'gpt-4o-mini'],
        index=['gemini-2.0-flash', 'gpt-4o-mini'].index(st.session_state.selected_model)
    )
    
    text2sql_model = st.selectbox(
        "Text2SQL Model:",
        ['gemini-2.0-flash', 'qwen2.5-3b-sft', 'gpt-4o-mini'],
        index=['gemini-2.0-flash', 'qwen2.5-3b-sft', 'gpt-4o-mini'].index(st.session_state.selected_text2sql_model)
    )

    if chat_model != st.session_state.chat_model or text2sql_model != st.session_state.text2sql_model:
        st.session_state.chat_model = chat_model
        st.session_state.text2sql_model = text2sql_model
        chatbot = initialize(user_name, text2sql_model=str(st.session_state.text2sql_model), chat_model=str(st.session_state.chat_model))
        st.session_state.chatbot = chatbot


    

    with st.container():     
        if st.button("Clear Chat"):
            st.session_state.chatbot.create_new_chat(user_id=user_name)

    with st.chat_message( name="system"):
        st.markdown("© 2025 Nguyen Quang Hung. All rights reserved.")

    for message in st.session_state.chatbot.display_history:
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
        
        assistant_message = st.chat_message("assistant", avatar='graphics/assistant.png').empty()   
        
        streamed_text = ""
        for chunk in st.session_state.chatbot.stream(input_text):
            if isinstance(chunk, str):
                streamed_text += chunk
                assistant_message.write(streamed_text)
                
    st.write("Provide feedback on the response:")
    feedback = st.radio(
        "Did you find this response helpful?",
        ("Like", "Dislike"),
        horizontal=True
    )
        
    if st.button("Submit Feedback"):
        st.session_state.chatbot.update_feedback(feedback)
        st.success("Feedback submitted!")
          
        

     

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