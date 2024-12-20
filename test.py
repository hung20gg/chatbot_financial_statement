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
    TEXT2SQL_SWEET_SPOT_CONFIG,
    TEXT2SQL_EXP_GEMINI_CONFIG,
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    HORIZONTAL_PROMPT_BASE,
    HORIZONTAL_PROMPT_UNIVERSAL
)

from ETL.dbmanager.setup import (
    DBConfig,
    BGE_VERTICAL_BASE_CONFIG,
    BGE_VERTICAL_UNIVERSAL_CONFIG,
    BGE_HORIZONTAL_BASE_CONFIG,
    TEI_HORIZONTAL_UNIVERSAL_CONFIG,
    setup_db
)

from langchain_huggingface import HuggingFaceEmbeddings
import json
import torch

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


if __name__ == "__main__":
    
    
    db_config = DBConfig(**TEI_HORIZONTAL_UNIVERSAL_CONFIG)
    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    text2sql_config = Text2SQLConfig(**TEXT2SQL_FAST_OPENAI_CONFIG)
    prompt_config = PromptConfig(**VERTICAL_PROMPT_UNIVERSAL)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs = {'device': device})
    # embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': device})
    # db_config.embedding = embedding_model
    logging.info('Finish setup embedding')
    
    db = setup_db(db_config)
    logging.info('Finish setup db')
    
    text2sql = Text2SQL(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2)
    logging.info('Finish setup text2sql')
    
    chatbot = Chatbot(config = chat_config, text2sql = text2sql)
    logging.info('Finish setup chatbot')
    
    
    for c in chatbot.stream("Amount of customer deposits in BIDV and Vietcombank in Q2 2023"):
        if isinstance(c, str):

            print(c, end = "")
    
    # print(db.search_return_df('total assets', 2, 'bank'))
    # logging.info('Test search return account')
    
    # print(db.vector_db_company.similarity_search('Vinamilk', 2))
    # logging.info('Test search company')
    
    # prompt = "Amount of customer deposits in BIDV and Vietcombank in Q2 2023"
    # his, err, tab = text2sql.solve(prompt)

    
    # for t in tab:
    #     print(t.table)
        
    # with open('history.json', 'w') as f:
    #     json.dump(text2sql.llm_responses, f, indent=4)