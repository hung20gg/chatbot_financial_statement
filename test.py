# from ETL import setup_db, setup_db_openai
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

from agent import Chatbot, Text2SQL
from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    GPT4O_MINI_CONFIG,
    GPT4O_CONFIG,
    TEXT2SQL_MEDIUM_GEMINI_CONFIG,
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_CONFIG,
    TEXT2SQL_EXP_GEMINI_CONFIG,
    TEXT2SQL_THINKING_GEMINI_CONFIG

)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    HORIZONTAL_PROMPT_BASE,
    HORIZONTAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI,
)

from ETL.dbmanager.setup import (
    DBConfig,
    BGE_VERTICAL_BASE_CONFIG,
    BGE_VERTICAL_UNIVERSAL_CONFIG,
    BGE_HORIZONTAL_BASE_CONFIG,
    TEI_VERTICAL_UNIVERSAL_CONFIG,
    OPENAI_VERTICAL_UNIVERSAL_CONFIG,
    setup_db
)

import os
from initialize import initialize_text2sql


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def test():

    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    # text2sql_config = TEXT2SQL_FAST_GEMINI_CONFIG
    # text2sql_config['sql_llm'] = 'llama3.2-3b-test'
    text2sql_config = TEXT2SQL_THINKING_GEMINI_CONFIG
    # text2sql_config['sql_example_top_k'] = 1
    # text2sql_config['company_top_k'] = 1
    # text2sql_config['account_top_k'] = 4
    prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI

    # try:
    if True:
        text2sql = initialize_text2sql(text2sql_config, prompt_config)
        
        chatbot = Chatbot(config = chat_config, text2sql = text2sql)
        logging.info('Finish setup chatbot')
        
        
        logging.info('Test find stock code similarity')
        print(text2sql.db.find_stock_code_similarity('Ngân hàng TMCP Ngoại Thương Việt Nam', 2))
        print(text2sql.db.vector_db_ratio.similarity_search('ROA', 2))
        
        logging.info('Test text2sql')
        prompt = "For the year 2023, what was the Return on Equity (ROE) for Vietcombank (VCB) and Techcombank (TCB)?"
        his, err, tab = text2sql.solve(prompt)
        last_reasoning = his[-1]['content']

        print('===== Reasoning =====')
        print(last_reasoning)
        print('===== Table =====')
        print(tab[-1].table)
        
        
    # except Exception as e:
    #     logging.error("Failed to setup chatbot")
    #     logging.error(e)


if __name__ == "__main__":
    
    
    test()