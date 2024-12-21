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
    OPENAI_VERTICAL_UNIVERSAL_CONFIG,
    setup_db
)

from langchain_huggingface import HuggingFaceEmbeddings
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from ETL.connector import check_embedding_server

def test():
    
    chat_config = ChatConfig(**GPT4O_MINI_CONFIG)
    text2sql_config = Text2SQLConfig(**TEXT2SQL_FASTEST_CONFIG)
    prompt_config = PromptConfig(**VERTICAL_PROMPT_UNIVERSAL)
    
    embedding_server = os.getenv('EMBEDDING_SERVER_URL')
    
    if check_embedding_server(embedding_server):
        logging.info('Using remote embedding server')
        db_config = DBConfig(**TEI_HORIZONTAL_UNIVERSAL_CONFIG)
    elif os.path.exists('data/vector_db_vertical_openai'):
        logging.info('Using openai embedding')
        db_config = DBConfig(**OPENAI_VERTICAL_UNIVERSAL_CONFIG)
    
    elif os.getenv('LOCAL_EMBEDDING'):
        import torch
    
        db_config = DBConfig(**BGE_VERTICAL_UNIVERSAL_CONFIG)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs = {'device': device})
        db_config.embedding = embedding_model
    
    else:
        raise ValueError('No Embedding Method Found')
    
    
    logging.info('Finish setup embedding')
    
    try:
        db = setup_db(db_config)
        logging.info('Finish setup db')
        
        text2sql = Text2SQL(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2)
        logging.info('Finish setup text2sql')
        
        chatbot = Chatbot(config = chat_config, text2sql = text2sql)
        logging.info('Finish setup chatbot')
        
        
        logging.info('Test find stock code similarity')
        print(db.find_stock_code_similarity('Ngân hàng TMCP Ngoại Thương Việt Nam', 2))
        print(db.vector_db_ratio.similarity_search('ROA', 2))
        
        logging.info('Test text2sql')
        prompt = "Amount of customer deposits in BIDV and Vietcombank in Q2 2023"
        his, err, tab = text2sql.solve(prompt)
        print(tab[-1].table)
        
    except Exception as e:
        logging.error("Failed to setup chatbot")
        logging.error(e)


if __name__ == "__main__":
    
    
    test()