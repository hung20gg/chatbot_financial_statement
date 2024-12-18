import os
import openai
import streamlit as st

import sys 
# sys.path.append("..")

from agent import Chatbot, Text2SQL
from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    GPT4O_MINI_CONFIG,
    GPT4O_CONFIG,
    TEXT2SQL_MEDIUM_OPENAI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_SWEET_SPOT_CONFIG
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
import json
# import torch

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


st.set_page_config(
    page_title="Chatbot",
    page_icon="graphics/Icon-BIDV.png" 
)

@st.cache_resource
def initialize():
    db_config = DBConfig(**BGE_VERTICAL_UNIVERSAL_CONFIG)
    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    text2sql_config = Text2SQLConfig(**TEXT2SQL_FAST_OPENAI_CONFIG)
    prompt_config = PromptConfig(**VERTICAL_PROMPT_UNIVERSAL)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': device})
    # db_config.embedding = embedding_model
    logging.info('Finish setup embedding')
    
    db = setup_db(db_config)
    logging.info('Finish setup db')
    
    text2sql = Text2SQL(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2)
    logging.info('Finish setup text2sql')
    
    chatbot = Chatbot(config = chat_config, text2sql = text2sql)
    logging.info('Finish setup chatbot')
    return chatbot

chatbot = initialize()


    
st.session_state.chatbot = chatbot

with st.container():     
    if st.button("Clear Chat"):
        st.session_state.chatbot.setup()

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
     
