from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pydantic import BaseModel, SkipValidation
from typing import List, Any

import re

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


class HubVerticalBase(BaseDBHUB):
    # Required from BaseDBHUB
    conn: SkipValidation
    vector_db_company: Chroma
    vector_db_sql: Chroma
    multi_threading: bool = False
    
    # Additional attributes
    vector_db_bank: Chroma 
    vector_db_non_bank: Chroma
    vector_db_securities: Chroma
    vector_db_ratio: Chroma
    
    def __init__(self, conn, 
                 vector_db_bank: Chroma, 
                 vector_db_non_bank: Chroma, 
                 vector_db_securities: Chroma, 
                 vector_db_ratio: Chroma, 
                 vector_db_company: Chroma, 
                 vector_db_sql: Chroma,
                 multi_threading = True): # Multi-thread only useful for online embedding
        
        super().__init__(
            conn=conn,
            vector_db_company=vector_db_company,
            vector_db_sql=vector_db_sql,
            multi_threading=multi_threading,
            
            vector_db_bank = vector_db_bank,
            vector_db_non_bank = vector_db_non_bank,
            vector_db_securities = vector_db_securities,
            vector_db_ratio = vector_db_ratio,
        )
        logging.info('Finish setup for Vertical Base')
        
        
    # ================== Search for suitable content (account) ================== #
    
    def _accounts_search(self, texts: List[str], top_k: int, type_, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]
        for text in texts:
            if type_ == 'bank':
                result = self.vector_db_bank.similarity_search(text, top_k)
            elif type_ == 'non_bank':
                result = self.vector_db_non_bank.similarity_search(text, top_k)    
            elif type_ == 'securities':
                result = self.vector_db_securities.similarity_search(text, top_k)
            elif type_ == 'ratio':
                result = self.vector_db_ratio.similarity_search(text, top_k)
            else:
                raise ValueError("Query table not supported")
            
            
            for item in result:
                try:
                    collect_code.add(item.metadata['code'])
                except Exception as e:
                    print(e)
        return list(collect_code)
    
    def _accounts_search_multithread(self, texts: List[str], top_k: int, type_:str, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]

        # Define a function for parallel execution
        def search_text(text):
            if type_ == 'bank':
                result = self.vector_db_bank.similarity_search(text, top_k)
            elif type_ == 'non_bank':
                result = self.vector_db_non_bank.similarity_search(text, top_k)
            elif type_ == 'securities':
                result = self.vector_db_securities.similarity_search(text, top_k)
            elif type_ == 'ratio':
                result = self.vector_db_ratio.similarity_search(text, top_k)
            else:
                raise ValueError("Query table not supported")
            # Extract the stock codes from the search result
            return [item.metadata['code'] for item in result]
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(search_text, texts)
            

        # Collect and combine results
        for codes in results:
            collect_code.update(codes)

        return list(collect_code)
    
    
    def search_return_df(self, texts, top_k, type_ = 'non_bank') -> pd.DataFrame:
        
        """
        Perform a search for the most similar account codes based on the provided text.
        
        Return the result as a DataFrame.
        """
        collect_code = self.accounts_search(texts, top_k, type_ = type_)
        # collect_code = [f"'{code}'" for code in collect_code]
        
        placeholder = ', '.join(['%s' for _ in collect_code])
        if type_ == 'ratio':
            query = f"SELECT ratio_code, ratio_name FROM map_category_code_ratio WHERE ratio_code IN ({placeholder})"
        else:
            query = f"SELECT category_code, en_caption FROM map_category_code_{type_} WHERE category_code IN ({placeholder})"
        
        return self.query(query, params=collect_code, return_type='dataframe')
    
    

    # ================== Search for suitable Mapping table ================== #
    
    def _return_mapping_table(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
        
        start = time.time()
        check_status_table = {
            'map_category_code_non_bank': True,
            'map_category_code_bank': True,
            'map_category_code_securities': True,
            'map_category_code_ratio': True
        }
        
        if len(stock_code) != 0 and not get_all_tables:
            company_df = self.return_company_from_stock_codes(stock_code)
            try:
                if company_df['is_bank'].sum() == 0:
                    check_status_table['map_category_code_bank'] = False
                if company_df['is_securities'].sum() == 0:
                    check_status_table['map_category_code_securities'] = False
                if company_df['is_bank'].sum() + company_df['is_securities'].sum() == len(company_df):
                    check_status_table['map_category_code_non_bank'] = False  
            except Exception as e:
                print(e)
                pass   
         
        # Avoid override from the previous check
        if len(industry) != 0 and not get_all_tables:
            exact_industries = self.__get_exact_industry_bm25(industry)
            for ind in exact_industries:
                if ind == 'Banking':
                    check_status_table['map_category_code_non_bank'] = True
                if ind == 'Financial Services':
                    check_status_table['map_category_code_securities'] = True
                else:
                    check_status_table['map_category_code_bank'] = True
                
        return_table = {
            'map_category_code_non_bank': None,
            'map_category_code_bank': None,
            'map_category_code_securities': None,
            'map_category_code_ratio': None
        }        
                
        if len(financial_statement_row) != 0:  
            if check_status_table['map_category_code_non_bank']:
                return_table['map_category_code_non_bank'] = self.search_return_df(financial_statement_row, top_k, type_='non_bank')
            if check_status_table['map_category_code_bank']:
                return_table['map_category_code_bank'] = self.search_return_df(financial_statement_row, top_k, type_='bank')
            if check_status_table['map_category_code_securities']:
                return_table['map_category_code_securities'] = self.search_return_df(financial_statement_row, top_k, type_='securities')
                
        if len(financial_ratio_row) != 0:
            return_table['map_category_code_ratio'] = self.search_return_df(financial_ratio_row, top_k, type_='ratio')
           
        end = time.time()
        logging.info(f"Time taken to return mapping table: {end-start}") 
        return return_table


    def _return_mapping_table_multithread(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
                
        start = time.time()
        
        check_status_table = {
            'map_category_code_non_bank': True,
            'map_category_code_bank': True,
            'map_category_code_securities': True,
            'map_category_code_ratio': True
        }
        
        if len(stock_code) != 0 and not get_all_tables:
            company_df = self.return_company_from_stock_codes(stock_code)
            try:
                if company_df['is_bank'].sum() == 0:
                    check_status_table['map_category_code_bank'] = False
                if company_df['is_securities'].sum() == 0:
                    check_status_table['map_category_code_securities'] = False
                if company_df['is_bank'].sum() + company_df['is_securities'].sum() == len(company_df):
                    check_status_table['map_category_code_non_bank'] = False  
            except Exception as e:
                print(e)
                pass   
         
        # Avoid override from the previous check
        if len(industry) != 0 and not get_all_tables:
            exact_industries = self.__get_exact_industry_bm25(industry)
            for ind in exact_industries:
                if ind == 'Banking':
                    check_status_table['map_category_code_non_bank'] = True
                if ind == 'Financial Services':
                    check_status_table['map_category_code_securities'] = True
                else:
                    check_status_table['map_category_code_bank'] = True
                
        return_table = {
            'map_category_code_non_bank': None,
            'map_category_code_bank': None,
            'map_category_code_securities': None,
            'map_category_code_ratio': None
        }   
        
        tasks = []     
                
        if len(financial_statement_row) != 0:  
            if check_status_table['map_category_code_non_bank']:
                tasks.append(('map_category_code_non_bank', financial_statement_row, top_k, 'non_bank'))
                
            if check_status_table['map_category_code_bank']:
                tasks.append(('map_category_code_bank', financial_statement_row, top_k, 'bank'))
                
            if check_status_table['map_category_code_securities']:
                tasks.append(('map_category_code_securities', financial_statement_row, top_k, 'securities'))
                
        if len(financial_ratio_row) != 0:
            tasks.append(('map_category_code_ratio', financial_ratio_row, top_k, 'ratio'))
            
        def process_task(task):
            table_name, financial_statement_row, top_k, type_ = task
            return table_name, self.search_return_df(financial_statement_row, top_k, type_)
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_task, tasks)
            
        for table_name, result in results:
            return_table[table_name] = result
            
        end = time.time()
        logging.info(f"Time taken to return mapping table multithread: {end-start}")     
        return return_table






class HubVerticalUniversal(BaseDBHUB):
    
    # Required from BaseDBHUB
    conn: SkipValidation
    vector_db_company: Chroma
    vector_db_sql: Chroma
    multi_threading: bool = False
    
    vector_db_ratio : Chroma
    vector_db_fs : Chroma
    
    def __init__(self, conn, 
                 vector_db_ratio: Chroma, 
                 vector_db_fs: Chroma, 
                 vector_db_company: Chroma, 
                 vector_db_sql: Chroma,
                 multi_threading = True):
        super().__init__(
            conn=conn,
            vector_db_company=vector_db_company,
            vector_db_sql=vector_db_sql,
            multi_threading=multi_threading,
            
            vector_db_ratio = vector_db_ratio,
            vector_db_fs = vector_db_fs
        )
        logging.info('Finish setup for Vertical Universal')
        
    # ================== Search for suitable content (account) ================== #
    def _accounts_search(self, texts, top_k, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]
            
        for text in texts:
            result = self.vector_db_fs.similarity_search(text, top_k)
            
            for item in result:
                try:
                    collect_code.add(item.metadata['code'])
                except Exception as e:
                    print(e)
        return list(collect_code)
    
    
    def _accounts_search_multithread(self, texts, top_k, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]

        # Define a function for parallel execution
        def search_text(text):
            result = self.vector_db_fs.similarity_search(text, top_k)
            return [item.metadata['code'] for item in result]
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(search_text, texts)
            

        # Collect and combine results
        for codes in results:
            collect_code.update(codes)

        return list(collect_code)
    
    def search_return_df(self, texts, top_k, type_ = None, **kwargs) -> pd.DataFrame:
        
        """
        Perform a search for the most similar account codes based on the provided text.
        
        Return the result as a DataFrame.
        """
        collect_code = self.accounts_search(texts, top_k)
        
        placeholder = ', '.join(['%s' for _ in collect_code])
        query = f"SELECT universal_code, universal_caption FROM map_category_code_universal WHERE universal_code IN ({placeholder})"
        
        return self.query(query, params=collect_code)
    
    # ================== Search for suitable Mapping table ================== #
    
    def _return_mapping_table(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
        
        start = time.time()
        
        return_table = {
            'map_category_code_universal': None,
            'map_category_code_ratio': None
        }        
                
        if len(financial_statement_row) != 0:  
            return_table['map_category_code_universal'] = self.search_return_df(financial_statement_row, top_k)
                
        if len(financial_ratio_row) != 0:
            return_table['map_category_code_ratio'] = self.search_return_df(financial_ratio_row, top_k)
           
        end = time.time()
        logging.info(f"Time taken to return mapping table: {end-start}") 
        return return_table
    
    
    def _return_mapping_table_multithread(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
                
        start = time.time()
        
        return_table = {
            'map_category_code_universal': None,
            'map_category_code_ratio': None
        }   
        
        tasks = []     
                
        if len(financial_statement_row) != 0:  
            tasks.append(('map_category_code_universal', financial_statement_row, top_k))
                
        if len(financial_ratio_row) != 0:
            tasks.append(('map_category_code_ratio', financial_ratio_row, top_k))
            
        def process_task(task):
            table_name, financial_statement_row, top_k = task
            return table_name, self.search_return_df(financial_statement_row, top_k)
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_task, tasks)
            
        for table_name, result in results:
            return_table[table_name] = result
            
        end = time.time()
        logging.info(f"Time taken to return mapping table multithread: {end-start}")     
        return return_table