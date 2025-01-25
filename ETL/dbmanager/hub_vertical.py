from langchain_core.vectorstores import VectorStore
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
from .rerank import BaseRerannk


class HubVerticalBase(BaseDBHUB):

    # Additional attributes
    vector_db_bank: VectorStore 
    vector_db_non_bank: VectorStore
    vector_db_securities: VectorStore
    vector_db_ratio: VectorStore
    
    hub_name: str = "HubVerticalBase"
        
        
    # ================== Search for suitable content (account) ================== #
    
    def _accounts_search(self, texts: List[str], top_k: int, type_, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]
        for text in texts:
            if type_ == 'bank':
                result = self._similarity_search(self.vector_db_bank, text, top_k)
                # result = self.vector_db_bank.similarity_search(text, top_k)
            elif type_ == 'non_bank':
                result = self._similarity_search(self.vector_db_non_bank, text, top_k)
                # result = self.vector_db_non_bank.similarity_search(text, top_k)    
            elif type_ == 'securities':
                result = self._similarity_search(self.vector_db_securities, text, top_k)
                # result = self.vector_db_securities.similarity_search(text, top_k)
            elif type_ == 'ratio':
                result = self._similarity_search(self.vector_db_ratio, text, top_k)
                # result = self.vector_db_ratio.similarity_search(text, top_k)
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
                result = self._similarity_search(self.vector_db_bank, text, top_k)
                # result = self.vector_db_bank.similarity_search(text, top_k)
            elif type_ == 'non_bank':
                result = self._similarity_search(self.vector_db_non_bank, text, top_k)
                # result = self.vector_db_non_bank.similarity_search(text, top_k)    
            elif type_ == 'securities':
                result = self._similarity_search(self.vector_db_securities, text, top_k)
                # result = self.vector_db_securities.similarity_search(text, top_k)
            elif type_ == 'ratio':
                result = self._similarity_search(self.vector_db_ratio, text, top_k)
                # result = self.vector_db_ratio.similarity_search(text, top_k)
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

            return self.query(query, params=collect_code, return_type='dataframe')
        
        else: # category_code in explaination and financial statement are come from same vector db
            
            collect_code_fs = [code for code in collect_code if 'TM' not in code]
            collect_code_tm = [code for code in collect_code if 'TM' in code]
            
            placeholder_fs = ', '.join(['%s' for _ in collect_code_fs])
            placeholder_tm = ', '.join(['%s' for _ in collect_code_tm])
            
            dfs = []
            
            if len(collect_code_fs) != 0:
                query = f"SELECT category_code, en_caption FROM map_category_code_{type_} WHERE category_code IN ({placeholder_fs})"
            
                df = self.query(query, params=collect_code_fs, return_type='dataframe')
                dfs.append(df)
                
            if len(collect_code_tm) != 0:
                query = f"SELECT category_code, en_caption FROM map_category_code_explaination_{type_} WHERE category_code IN ({placeholder_tm})"
                df_tm = self.query(query, params=collect_code_tm, return_type='dataframe') 
                
                dfs.append(df_tm)
            df = pd.concat(dfs)
            
            return df
                
                
    
    

    # ================== Search for suitable Mapping table ================== #
    
    def _return_mapping_table(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
        
        start = time.time()
        check_status_table = {
            'category_code_non_bank': True,
            'category_code_bank': True,
            'category_code_securities': True,
            'category_code_ratio': True
        }
        
        if len(stock_code) != 0 and not get_all_tables:
            company_df = self.return_company_from_stock_codes(stock_code)
            try:
                if company_df['is_bank'].sum() == 0:
                    check_status_table['category_code_bank'] = False
                if company_df['is_securities'].sum() == 0:
                    check_status_table['category_code_securities'] = False
                if company_df['is_bank'].sum() + company_df['is_securities'].sum() == len(company_df):
                    check_status_table['category_code_non_bank'] = False  
            except Exception as e:
                print(e)
                pass   
         
        # Avoid override from the previous check
        if len(industry) != 0 and not get_all_tables:
            exact_industries = self.get_exact_industry_bm25(industry)
            for ind in exact_industries:
                if ind == 'Banking':
                    check_status_table['category_code_non_bank'] = True
                if ind == 'Financial Services':
                    check_status_table['category_code_securities'] = True
                else:
                    check_status_table['category_code_bank'] = True
                
        return_table = {
            'category_code_non_bank': None,
            'category_code_bank': None,
            'category_code_securities': None,
            'category_code_ratio': None
        }    

        if len(industry) != 0:
            exact_industries = self.get_exact_industry_bm25(industry)
            df_industry = pd.DataFrame(exact_industries, columns=['industry'])
            return_table['industry'] = df_industry
                
        if len(financial_statement_row) != 0:  
            if check_status_table['category_code_non_bank']:
                return_table['category_code_non_bank'] = self.search_return_df(financial_statement_row, top_k, type_='non_bank')
            if check_status_table['category_code_bank']:
                return_table['category_code_bank'] = self.search_return_df(financial_statement_row, top_k, type_='bank')
            if check_status_table['category_code_securities']:
                return_table['category_code_securities'] = self.search_return_df(financial_statement_row, top_k, type_='securities')
                
        if len(financial_ratio_row) != 0:
            return_table['category_code_ratio'] = self.search_return_df(financial_ratio_row, top_k, type_='ratio')
           
        end = time.time()
        logging.info(f"Time taken to return mapping table: {end-start}") 
        return return_table


    def _return_mapping_table_multithread(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
                
        start = time.time()
        
        check_status_table = {
            'category_code_non_bank': True,
            'category_code_bank': True,
            'category_code_securities': True,
            'category_code_ratio': True
        }
        
        if len(stock_code) != 0 and not get_all_tables:
            company_df = self.return_company_from_stock_codes(stock_code)
            try:
                if company_df['is_bank'].sum() == 0:
                    check_status_table['category_code_bank'] = False
                if company_df['is_securities'].sum() == 0:
                    check_status_table['category_code_securities'] = False
                if company_df['is_bank'].sum() + company_df['is_securities'].sum() == len(company_df):
                    check_status_table['category_code_non_bank'] = False  
            except Exception as e:
                print(e)
                pass   
         
        # Avoid override from the previous check
        if len(industry) != 0 and not get_all_tables:
            exact_industries = self.get_exact_industry_bm25(industry)
            for ind in exact_industries:
                if ind == 'Banking':
                    check_status_table['category_code_non_bank'] = True
                if ind == 'Financial Services':
                    check_status_table['category_code_securities'] = True
                else:
                    check_status_table['category_code_bank'] = True
                
        return_table = {
            'category_code_non_bank': None,
            'category_code_bank': None,
            'category_code_securities': None,
            'category_code_ratio': None
        }   

        if len(industry) != 0:
            exact_industries = self.get_exact_industry_bm25(industry)
            df_industry = pd.DataFrame(exact_industries, columns=['industry'])
            return_table['industry'] = df_industry
        
        tasks = []     
                
        if len(financial_statement_row) != 0:  
            if check_status_table['category_code_non_bank']:
                tasks.append(('category_code_non_bank', financial_statement_row, top_k, 'non_bank'))
                
            if check_status_table['category_code_bank']:
                tasks.append(('category_code_bank', financial_statement_row, top_k, 'bank'))
                
            if check_status_table['category_code_securities']:
                tasks.append(('category_code_securities', financial_statement_row, top_k, 'securities'))
                
        if len(financial_ratio_row) != 0:
            tasks.append(('category_code_ratio', financial_ratio_row, top_k, 'ratio'))
            
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
    
    vector_db_ratio : VectorStore
    vector_db_fs : VectorStore
    
    hub_name: str = "HubVerticalUniversal"
        
    # ================== Search for suitable content (account) ================== #
    def _accounts_search(self, texts, top_k, type_ = None, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]
            
        for text in texts:
            if type_ == 'ratio':
                result = self._similarity_search(self.vector_db_ratio, text, top_k)
                # result = self.vector_db_ratio.similarity_search(text, top_k)
            else:
                result = self._similarity_search(self.vector_db_fs, text, top_k)
                # result = self.vector_db_fs.similarity_search(text, top_k)
            
            for item in result:
                try:
                    collect_code.add(item.metadata['code'])
                except Exception as e:
                    print(e)
        return list(collect_code)
    
    
    def _accounts_search_multithread(self, texts, top_k, type_ = None, **kwargs):
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]

        # Define a function for parallel execution
        def search_text(text):
            if type_ == 'ratio':
                result = self._similarity_search(self.vector_db_ratio, text, top_k)
                # result = self.vector_db_ratio.similarity_search(text, top_k)
            else:
                result = self._similarity_search(self.vector_db_fs, text, top_k)
                # result = self.vector_db_fs.similarity_search(text, top_k)
            
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
        collect_code = self.accounts_search(texts, top_k, type_ = type_)
        print(collect_code)
        
        collect_code_fs = [code for code in collect_code if 'TM' not in code]
        collect_code_tm = [code for code in collect_code if 'TM' in code]
        
        print(collect_code_fs)
        print(collect_code_tm)
        
        collect_code_fs = [code for code in collect_code if 'TM' not in code]
        collect_code_tm = [code for code in collect_code if 'TM' in code]
        
        placeholder = ', '.join(['%s' for _ in collect_code])
        placeholder_fs = ', '.join(['%s' for _ in collect_code_fs])
        placeholder_tm = ', '.join(['%s' for _ in collect_code_tm])
        
        
        
        if type_ == 'ratio':
            query = f"SELECT ratio_code, ratio_name FROM map_category_code_ratio WHERE ratio_code IN ({placeholder})"
            
            return self.query(query, params=collect_code) 
        
        else:
            dfs = []
            
            if len(collect_code_fs) != 0:
                query = f"SELECT category_code, en_caption FROM map_category_code_universal WHERE category_code IN ({placeholder_fs})"
                df = self.query(query, params=collect_code_fs) 
                dfs.append(df)
            
            if len(collect_code_tm) != 0:
                query = f"SELECT category_code, en_caption FROM map_category_code_explaination WHERE category_code IN ({placeholder_tm})"
                df_tm = self.query(query, params=collect_code_tm) 
                
                dfs.append(df_tm)
            df = pd.concat(dfs)
            return df
    
    # ================== Search for suitable Mapping table ================== #
    
    def _return_mapping_table(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
        
        start = time.time()
        
        return_table = {
            'category_code_universal': None,
            'category_code_ratio': None
        }     

        if len(industry) != 0:
            exact_industries = self.get_exact_industry_bm25(industry)
            df_industry = pd.DataFrame(exact_industries, columns=['industry'])
            return_table['industry'] = df_industry   
                
        if len(financial_statement_row) != 0:  
            return_table['category_code_universal'] = self.search_return_df(financial_statement_row, top_k, type_='fs')
                
        if len(financial_ratio_row) != 0:
            return_table['category_code_ratio'] = self.search_return_df(financial_ratio_row, top_k, type_='ratio')
           
        end = time.time()
        logging.info(f"Time taken to return mapping table: {end-start}") 
        return return_table
    
    
    def _return_mapping_table_multithread(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
                
        start = time.time()
        
        return_table = {
            'category_code_universal': None,
            'category_code_ratio': None
        }   
        
        if len(industry) != 0:
            exact_industries = self.get_exact_industry_bm25(industry)
            df_industry = pd.DataFrame(exact_industries, columns=['industry'])
            return_table['industry'] = df_industry

        tasks = []     
                
        if len(financial_statement_row) != 0:  
            tasks.append(('category_code_universal', financial_statement_row, top_k, 'fs'))
                
        if len(financial_ratio_row) != 0:
            tasks.append(('category_code_ratio', financial_ratio_row, top_k, 'ratio'))
            
        def process_task(task):
            table_name, financial_statement_row, top_k, type_ = task
            return table_name, self.search_return_df(financial_statement_row, top_k, type_=type_)
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_task, tasks)
            
        for table_name, result in results:
            return_table[table_name] = result
            
        end = time.time()
        logging.info(f"Time taken to return mapping table multithread: {end-start}")     
        return return_table