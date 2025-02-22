import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()
import sys 
sys.path.append('..')

from agent import Chatbot, Text2SQL
from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    GPT4O_MINI_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,

)

from agent.prompt.prompt_controller import (
    PromptConfig, 

    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI,

)


import os
from initialize import initialize_text2sql


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_text2sql_chat(version = 'v3', enhance = False):

    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    text2sql_config = TEXT2SQL_FAST_GEMINI_CONFIG
    prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI

    text2sql = initialize_text2sql(text2sql_config, prompt_config, version=version, message=True)

    prompt = "ROAA of banking industry from 2016 to 2023"

    generator = text2sql.stream(prompt, adjust_table = 'text', enhance = enhance)

    for text in generator:
        print(text)


if __name__ == "__main__":
    version = 'v3.2'
    enhance = 'correction'
    test_text2sql_chat(version, enhance=enhance)