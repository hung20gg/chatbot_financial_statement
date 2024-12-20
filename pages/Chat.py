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
    TEXT2SQL_MEDIUM_OPENAI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_SWEET_SPOT_CONFIG,
    TEXT2SQL_4O_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
)

from ETL.dbmanager.setup import (
    DBConfig,
    BGE_VERTICAL_BASE_CONFIG,
    BGE_VERTICAL_UNIVERSAL_CONFIG,
    OPENAI_VERTICAL_UNIVERSAL_CONFIG,
    setup_db
)

from langchain_huggingface import HuggingFaceEmbeddings
from ETL.dbmanager import get_semantic_layer
import json
import torch

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
def initialize(user_name):
    db_config = DBConfig(**OPENAI_VERTICAL_UNIVERSAL_CONFIG)
    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    text2sql_config = Text2SQLConfig(**TEXT2SQL_FAST_OPENAI_CONFIG)
    prompt_config = PromptConfig(**VERTICAL_PROMPT_UNIVERSAL)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs = {'device': device})
    # # embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': device})    db_config.embedding = embedding_model
    # db_config.embedding = embedding_model
    
    logging.info('Finish setup embedding')
    
    db = setup_db(db_config)
    logging.info('Finish setup db')
    
    text2sql = Text2SQL(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2)
    logging.info('Finish setup text2sql')
    
    message_saver = get_semantic_layer()
    
    chatbot = ChatbotSematic(config = chat_config, text2sql = text2sql, message_saver = message_saver)
    logging.info('Finish setup chatbot')
    
    chatbot.create_new_chat(user_id=user_name)
    
    
    return chatbot
def chat(user_name):
    user_name = str(user_name)
    
    chatbot = initialize(user_name)
    
    st.session_state.chatbot = chatbot

    with st.container():     
        if st.button("Clear Chat"):
            st.session_state.chatbot.create_new_chat(user_id=user_name)

    with st.chat_message( name="system"):
        st.markdown("© 2024 Nguyen Quang Hung. All rights reserved.")

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