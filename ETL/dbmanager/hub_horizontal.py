from .hub_vertical import HubVerticalBase, HubVerticalUniversal
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pydantic import BaseModel, SkipValidation
from typing import List, Any

from dotenv import load_dotenv
import os

load_dotenv()

import logging
import pandas as pd
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from ..connector import *
from .abstracthub import BaseDBHUB

class HubHorizontalBase(HubVerticalBase):
    # def __init__(self, conn, 
    #              vector_db_bank: Chroma, 
    #              vector_db_non_bank: Chroma, 
    #              vector_db_securities: Chroma, 
    #              vector_db_ratio: Chroma, 
    #              vector_db_company: Chroma, 
    #              vector_db_sql: Chroma,
    #              multi_threading = True):
    #     super().__init__(
    #         conn=conn,
    #         vector_db_bank=vector_db_bank,
    #         vector_db_non_bank=vector_db_non_bank,
    #         vector_db_securities=vector_db_securities,
    #         vector_db_ratio=vector_db_ratio,
    #         vector_db_company=vector_db_company,
    #         vector_db_sql=vector_db_sql,
    #         multi_threading=multi_threading
    #     )
    #     logging.info('Finish setup for Horizontal Base')
    pass

class HubHorizontalUniversal(HubVerticalUniversal):
    def __init__(self, conn, 
                 vector_db_ratio: Chroma, 
                 vector_db_fs: Chroma, 
                 vector_db_company: Chroma, 
                 vector_db_sql: Chroma,
                 multi_threading = True):
        super().__init__(
            conn=conn,
            vector_db_ratio=vector_db_ratio,
            vector_db_fs=vector_db_fs,
            vector_db_company=vector_db_company,
            vector_db_sql=vector_db_sql,
            multi_threading=multi_threading
        )
        logging.info('Finish setup for Horizontal Universal')