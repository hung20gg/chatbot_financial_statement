import sys 
# sys.path.append('..')
from pydantic import BaseModel, SkipValidation, ConfigDict
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from ..connector import *

from dotenv import load_dotenv
import os

load_dotenv()

import pandas as pd
import time

class BaseDBHUB(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conn: SkipValidation
    vector_db_company: Chroma
    vector_db_sql: Chroma
    multi_threading: bool = False
        
    def rasie_multi_threading_error(self):
        if not self.multi_threading:
            raise Exception("This method is not supported in multi threading mode.")

    
    # ================== Search for suitable content (account) ================== #
    
    def _accounts_search(self, texts, top_k, **kwargs) -> list[str]:
        """ 
        An abstract method that must be implemented by subclasses. It is intended to search for 
        suitable accounts based on the provided texts.
        """
        raise NotImplementedError("Subclasses must implement the 'search' method.")

    
    def _accounts_search_multithread(self, texts, top_k, **kwargs):
        """ 
        An abstract method perform multithreaded search for suitable accounts based on the provided texts.
        """
        raise NotImplementedError("Subclasses must implement the 'search_multithread' method.")


    def accounts_search(self, texts, top_k, **kwargs) -> list[str]:        
        if self.multi_threading:
            return self._accounts_search_multithread(texts, top_k, **kwargs)
        else:
            return self._accounts_search(texts, top_k, **kwargs)
        
        
    
    def search_return_df(self, texts, top_k, **kwargs):
        """ 
        An abstract method that must be implemented by subclasses. It is intended to search for suitable 
        accounts and return the results as a DataFrame.
        """
        raise NotImplementedError("Subclasses must implement the 'search_return_df' method.")
    
   
    # ================== Query ================== #
    def query(self, query, **kwargs) -> pd.DataFrame|str:
        """ 
        Execute a SQL query and return the results as a DataFrame or a string if encountered an error.
        """
        return execute_query(query=query, conn=self.conn, **kwargs)
    
    # ================== Company Name to Stock Code ================== #
    
    ### Find stock code similarity using company name
    
    def __find_stock_code_similarity(self, company_name, top_k) -> list[str]:
        
        """
        The __find_stock_code_similarityd method performs a multi-threaded search to find stock codes 
        that are similar to the provided company names. 
        
        It uses a vector database to perform similarity searches and returns a list of matching stock 
        codes.
        """
        
        start = time.time()
        if isinstance(company_name, str):
            company_name = [company_name]
        stock_codes = set()
        for name in company_name:
            result = self.vector_db_company.similarity_search(name, top_k)
            for item in result:
                stock_codes.add(item.metadata['stock_code'])
        
        end = time.time()
        logging.info(f"Time taken to find stock code similarity: {end-start}")
        return list(stock_codes)
    
    def __find_stock_code_similarity_multithread(self, company_name, top_k) -> list[str]:
        
        """ 
        Perform a multi-threaded search to find stock codes that are similar to the provided company names.
        """
        
        if isinstance(company_name, str):
            company_name = [company_name]
        stock_codes = set()
        
        def search_name(name):
            result = self.vector_db_company.similarity_search(name, top_k)
            return [item.metadata['stock_code'] for item in result]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(search_name, company_name)
        
        for codes in results:
            stock_codes.update(codes)
            
        return list(stock_codes)
    
    def find_stock_code_similarity(self, company_name, top_k=2) -> list[str]:
        
        
        
        
        if self.multi_threading:
            return self.__find_stock_code_similarity_multithread(company_name, top_k)
        else:
            return self.__find_stock_code_similarity(company_name, top_k)
        
        
    ### Return exact stock code from company name
    
    def return_company_from_stock_codes(self, stock_codes) -> pd.DataFrame:
        
        """ 
        Returns company information based on the provided stock codes.
        """
        
        if not isinstance(stock_codes, list):
            stock_codes = [stock_codes]
        
        # If no stock code found
        if len(stock_codes) == 0:
            return pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        
        placeholder = ', '.join(['%s' for _ in stock_codes])
        query = f"SELECT stock_code, company_name, en_company_name, industry, is_bank, is_securities FROM company_info WHERE stock_code IN ({placeholder});"
        result = self.query(query, params=stock_codes)
        
        if isinstance(result, str):
            result = pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        return result
    
    
    def return_company_info(self, company_name, top_k=2) -> pd.DataFrame:
        stock_codes = self.find_stock_code_similarity(company_name, top_k)
        
        df = self.return_company_from_stock_codes(stock_codes)
        df.drop_duplicates(subset=['stock_code'], inplace=True)
        return df
    
    
    # ===== Find SQL query for few shot learning ===== #
    
    def find_sql_query(self, text, top_k=1):
        results = self.vector_db_sql.similarity_search(text, top_k)
        
        few_shot = ""
        for result in results:
            if result.metadata.get('sql_code', None) is not None:
                few_shot += '#### '+result.page_content + '\n\n'
                few_shot += f"```sql\n\n{result.metadata['sql_code']}```"
                
        return few_shot
    
    
    def get_exact_industry_bm25(self, industries):
        query = """
        SELECT distinct (industry)
        FROM company_info
        WHERE industry_tsvector @@ plainto_tsquery('english', '{industry}')
        LIMIT 50;
        """
        if not isinstance(industries, list):
            industries = [industries]
        exact_industries = set()
        for industry in industries:
            df = self.query(query.format(industry=industry))
            result = df['industry'].values.tolist()
            for item in result:
                exact_industries.add(item)
        return list(exact_industries)
    
    
    # ================== Search for suitable Mapping table ================== #

    
    def _return_mapping_table(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the 'return_mapping_table' method.")
            
            
    def _return_mapping_table_multithread(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the '_return_mapping_table_multithred' method.")
    
    def return_mapping_table(self, **kwargs):
        if self.multi_threading:
            return self._return_mapping_table_multithread(**kwargs)
        
        else:
            return self._return_mapping_table(**kwargs)