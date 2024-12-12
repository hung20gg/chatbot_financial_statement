from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Union
from langchain_huggingface import HuggingFaceEmbeddings

import os

from ..connector import *
from .hub_horizontal import HubHorizontalBase, HubHorizontalUniversal
from .hub_vertical import HubVerticalBase, HubVerticalUniversal

load_dotenv()


class DBConfig(BaseModel):
    embedding : Union[str, HuggingFaceEmbeddings]
    database_choice: str
    
    
OPENAI_VERTICAL_BASE_CONFIG = {
    "embedding": 'text-embedding-3-small',
    "database_choice": 'vertical_base'
}

OPENAI_VERTICAL_UNIVERSAL_CONFIG = {
    "embedding": 'text-embedding-3-small',
    "database_choice": 'vertical_universal'
}

OPENAI_HORIZONTAL_BASE_CONFIG = {
    "embedding": 'text-embedding-3-small',
    "database_choice": 'horizontal_base'
}

OPENAI_HORIZONTAL_UNIVERSAL_CONFIG = {
    "embedding": 'text-embedding-3-small',
    "database_choice": 'horizontal_universal'
}

BGE_VERTICAL_BASE_CONFIG = {
    "embedding": 'BAAI/bge-small-en-v1.5',
    "database_choice": 'vertical_base'
}

BGE_VERTICAL_UNIVERSAL_CONFIG = {
    "embedding": 'BAAI/bge-small-en-v1.5',
    "database_choice": 'vertical_universal'
}

BGE_HORIZONTAL_BASE_CONFIG = {
    "embedding": 'BAAI/bge-small-en-v1.5',
    "database_choice": 'horizontal_base'
}

BGE_HORIZONTAL_UNIVERSAL_CONFIG = {
    "embedding": 'BAAI/bge-small-en-v1.5',
    "database_choice": 'horizontal_universal'
}


def setup_db(config: DBConfig, multi_thread = True):
    conn = {
        'db_name': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    print(conn['db_name'])
    
    # conn = connect_to_db(**conn)
    current_directory = os.path.dirname(__file__)
    
    local_model = False
    if not isinstance(config.embedding, str):
        local_model = True
    elif config.embedding != 'text-embedding-3-small':
        local_model = True
    
    collection_chromadb = 'category_bank_chroma'
    persist_directory = os.path.join(current_directory, f'../../data/category_bank_chroma_{"local" if local_model else "openai"}')
    bank_vector_store = create_chroma_db(collection_chromadb, persist_directory, config.embedding)
    
    collection_chromadb = 'category_non_bank_chroma'
    persist_directory = os.path.join(current_directory, f'../../data/category_non_bank_chroma_{"local" if local_model else "openai"}')
    none_bank_vector_store = create_chroma_db(collection_chromadb, persist_directory, config.embedding)
    
    collection_chromadb = 'category_sec_chroma'
    persist_directory = os.path.join(current_directory, f'../../data/category_sec_chroma_{"local" if local_model else "openai"}')
    sec_vector_store = create_chroma_db(collection_chromadb, persist_directory, config.embedding)
    
    collection_chromadb = 'category_ratio_chroma'
    persist_directory = os.path.join(current_directory, f'../../data/category_ratio_chroma_{"local" if local_model else "openai"}')
    ratio_vector_store = create_chroma_db(collection_chromadb, persist_directory, config.embedding)
    
    collection_chromadb = 'company_name_chroma'
    persist_directory = os.path.join(current_directory, f'../../data/company_name_chroma_{"local" if local_model else "openai"}')
    vector_db_company = create_chroma_db(collection_chromadb, persist_directory, config.embedding)
    
    collection_chromadb = 'sql_query'
    persist_directory = os.path.join(current_directory, f'../../data/sql_query_{"local" if local_model else "openai"}')
    vector_db_sql = create_chroma_db(collection_chromadb, persist_directory, config.embedding)
    
    if config.database_choice == 'vertical_base':
        return HubVerticalBase(conn, 
                                 bank_vector_store, 
                                 none_bank_vector_store, 
                                 sec_vector_store, 
                                 ratio_vector_store, 
                                 vector_db_company, 
                                 vector_db_sql, 
                                 multi_thread)
        
    else:
        raise ValueError("Database choice not supported")