# from ETL import setup_db, setup_db_openai
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

from agent import Chatbot, Text2SQL
from agent.const import (
    ChatConfig,
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
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    HORIZONTAL_PROMPT_BASE,
    HORIZONTAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY_EXTEND,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI_EXTEND,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SHORT
)

import os
from initialize import initialize_text2sql


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def test(version = 'v3'):

    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    text2sql_config = TEXT2SQL_THINKING_GEMINI_CONFIG
    # text2sql_config['sql_llm'] = 'qwen2.5-3b-coder-test-v3/sft/v2.1'
    # text2sql_config['llm'] = 'qwen2.5-3b-coder-test-v3/sft/v2.1'
    # text2sql_config['sql_example_top_k'] = 0
    # # text2sql_config['company_top_k'] = 1
    # text2sql_config['account_top_k'] = 4
    prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI_EXTEND
    

    # try:
    if True:
        text2sql = initialize_text2sql(text2sql_config, prompt_config, version=version)
        
        chatbot = Chatbot(config = chat_config, text2sql = text2sql)
        logging.info('Finish setup chatbot')
        
        
        logging.info('Test find stock code similarity')
        print(text2sql.db.find_stock_code_similarity('Ngân hàng TMCP Ngoại Thương Việt Nam', 2))
        print(text2sql.db.vector_db_ratio.similarity_search('ROA', 2))
        
        logging.info('Test text2sql')
        # prompt = "For the year 2023, what was the Return on Equity of banking industry?"
        prompt = "Net Income QoQ of HPG in Q3 2024"
        output = text2sql.solve(prompt, adjust_table='text', mix_account=False, enhance='correction')

        print(output.extraction_msg)
        
        print('### ========= Reasoning ========= ###')
        for msg in output.history:
            print('\n# ===== Role: %s ===== #' % msg['role'])
            print('# ===== Content ===== #\n')
            print(msg['content'])

        print('===== Table =====')
        for t in output.execution_tables:
            print(t.table)
        
    # except Exception as e:
    #     logging.error("Failed to setup chatbot")
    #     logging.error(e)


if __name__ == "__main__":
    
    version = 'v3.2'
    test(version)