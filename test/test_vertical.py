import sys 
sys.path.append('..')

from ETL.dbmanager.setup import (
    DBConfig,
    BGE_VERTICAL_BASE_CONFIG,
    BGE_VERTICAL_UNIVERSAL_CONFIG,
    BGE_HORIZONTAL_BASE_CONFIG,
    BGE_HORIZONTAL_UNIVERSAL_CONFIG,
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
    db_config = DBConfig(**BGE_VERTICAL_BASE_CONFIG)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': device})
    db_config.embedding = embedding_model
    logging.info('Finish setup embedding')
    
    db = setup_db(db_config)
    logging.info('Finish setup db')
    
    print(db.find_stock_code_similarity('Ngân hàng TMCP Ngoại Thương Việt Nam', 2))
    logging.info('Test find stock code similarity')
    
    print(db.vector_db_fs.similarity_search('total assets', 2))
    
    print(db.search_return_df('total assets', 2, 'bank'))
    logging.info('Test search return account')
    
    # print(db.vector_db_sql.similarity_search('Compare the Return on Assets (ROA) and Return on Equity (ROE) of Vinamilk and Masan Group for the fiscal year 2023.  Additionally, provide the total assets and total equity for both companies for the same period.', 2))
    # logging.info('Test search SQL')