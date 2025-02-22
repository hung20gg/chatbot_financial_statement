from agent import Chatbot, Text2SQL, Text2SQLMessage
from agent.const import (
    ChatConfig,
    Text2SQLConfig,
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
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

from langchain_huggingface import HuggingFaceEmbeddings
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from ETL.connector import check_embedding_server
from ETL.dbmanager import get_semantic_layer, BaseRerannk



def initialize_text2sql(text2sql_config, prompt_config, version = 'v3', message = False, **kwargs) -> Text2SQL:
    text2sql_config = Text2SQLConfig(**text2sql_config)
    prompt_config = PromptConfig(**prompt_config)

    embedding_server = os.getenv('EMBEDDING_SERVER_URL')
    

    # Setup db
    if check_embedding_server(embedding_server):
        logging.info('Using remote embedding server')
        db_config = DBConfig(**TEI_VERTICAL_UNIVERSAL_CONFIG)
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

    reranker = BaseRerannk(name=os.getenv('RERANKER_SERVER_URL'))
    logging.info(f'Finish setup reranker, using reranker {reranker.reranker_type}')

    db = setup_db(db_config, reranker=reranker, version=version)
    logging.info('Finish setup db')

    if message:
        text2sql = Text2SQLMessage(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2, **kwargs)
    else:
        text2sql = Text2SQL(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2, **kwargs)
    logging.info('Finish setup text2sql')

    return text2sql