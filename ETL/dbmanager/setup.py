from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Union
from langchain_huggingface import HuggingFaceEmbeddings

from chromadb import Client, PersistentClient
from chromadb.config import Settings

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
        
    db_type = 'vertical' if 'vertical' in config.database_choice else 'horizontal'
    
    persist_client = PersistentClient(path = os.path.join(current_directory, f'../../data/vector_db_{db_type}_{"local" if local_model else "openai"}'), settings = Settings())
    
    if 'base' in config.database_choice:
        collection_chromadb = 'category_bank_chroma'
        bank_vector_store = create_chroma_db(collection_chromadb, persist_client, config.embedding)
        
        collection_chromadb = 'category_non_bank_chroma'
        none_bank_vector_store = create_chroma_db(collection_chromadb, persist_client, config.embedding)
        
        collection_chromadb = 'category_sec_chroma'
        sec_vector_store = create_chroma_db(collection_chromadb, persist_client, config.embedding)
    
        collection_chromadb = 'sql_query'
        vector_db_sql = create_chroma_db(collection_chromadb, persist_client, config.embedding)
    
    elif 'universal' in config.database_choice:
        collection_chromadb = 'category_universal_chroma'
        universal_vector_store = create_chroma_db(collection_chromadb, persist_client, config.embedding)
        
        collection_chromadb = 'sql_query_universal'
        vector_db_sql = create_chroma_db(collection_chromadb, persist_client, config.embedding)
    
    collection_chromadb = 'category_ratio_chroma'
    ratio_vector_store = create_chroma_db(collection_chromadb, persist_client, config.embedding)
    
    collection_chromadb = 'company_name_chroma'
    vector_db_company = create_chroma_db(collection_chromadb, persist_client, config.embedding)
    
    
    if config.database_choice == 'vertical_base':
        return HubVerticalBase(conn, 
                                 bank_vector_store, 
                                 none_bank_vector_store, 
                                 sec_vector_store, 
                                 ratio_vector_store, 
                                 vector_db_company, 
                                 vector_db_sql, 
                                 multi_thread)
        
    elif config.database_choice == 'vertical_universal':
        return HubVerticalUniversal(conn, 
                                     ratio_vector_store,
                                     universal_vector_store, 
                                     vector_db_company, 
                                     vector_db_sql, 
                                     multi_thread)
    elif config.database_choice == 'horizontal_base':
        return HubHorizontalBase(conn, 
                                 bank_vector_store, 
                                 none_bank_vector_store, 
                                 sec_vector_store, 
                                 ratio_vector_store, 
                                 vector_db_company, 
                                 vector_db_sql, 
                                 multi_thread)
        
    elif config.database_choice == 'horizontal_universal':
        return HubHorizontalUniversal(conn, 
                                      ratio_vector_store,
                                      universal_vector_store, 
                                      vector_db_company, 
                                      vector_db_sql, 
                                      multi_thread)
        
    else:
        raise ValueError("Database choice not supported")