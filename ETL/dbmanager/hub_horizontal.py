from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
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


class HubHorizontalBase(BaseDBHUB):
    
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
                 multi_thread = True): # Multi-thread only useful for online embedding
        
        super().__init__(conn, vector_db_company, vector_db_sql, multi_thread)
        self.vector_db_bank = vector_db_bank
        self.vector_db_non_bank = vector_db_non_bank
        self.vector_db_securities = vector_db_securities
        self.vector_db_ratio = vector_db_ratio
        
    # ================== Search for suitable columns ================== #
    
    def _columns_search(self, texts, top_k, type_, **kwargs):
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
    
    def _columns_search_multithread(self, texts, top_k, type_, **kwargs):
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
    
    
    def search_return_df(self, text, top_k, type_ = 'non_bank') -> pd.DataFrame:
        # print('search', text)
        if self.multi_thread:
            collect_code = self.search_multithread(text, top_k, type_)
        else:
            collect_code = self.search(text, top_k, type_)
        # print(collect_code)
        collect_code = [f"'{code}'" for code in collect_code]
        if type_ == 'ratio':
            query = f"SELECT ratio_code, ratio_name FROM map_category_code_ratio WHERE ratio_code IN ({', '.join(collect_code)})"
        else:
            query = f"SELECT category_code, en_caption FROM map_category_code_{type_} WHERE category_code IN ({', '.join(collect_code)})"
        return self.query(query,return_type='dataframe')
    
    
    def return_company_from_stock_codes(self, stock_codes):
        if not isinstance(stock_codes, list):
            stock_codes = [stock_codes]
        stock_codes = [f"'{code}'" for code in stock_codes]
        
        # If no stock code found
        if len(stock_codes) == 0:
            return pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        
        query = f"SELECT stock_code, company_name, en_company_name, industry, is_bank, is_securities FROM company_info WHERE stock_code IN ({', '.join(stock_codes)})"
        result = self.query(query, return_type='dataframe')
        if isinstance(result, str):
            result = pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        return result

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






class HubHorizontalUniversal(BaseDBHUB):
    vector_db_ratio : Chroma
    vector_db_fs : Chroma