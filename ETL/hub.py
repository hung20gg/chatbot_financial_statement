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

from .connector import *
# from connector import *
class DBHUB:
    """
    This will be the hub for both similarity search and rational DB
    A Centralized DB for all the queries
    """

    def __init__(self, conn, 
                 vector_db_bank: Chroma, 
                 vector_db_non_bank: Chroma, 
                 vector_db_securities: Chroma, 
                 vector_db_ratio: Chroma, 
                 vector_db_company: Chroma, 
                 vector_db_sql: Chroma,
                 multi_thread = True): # Multi-thread only useful for online embedding
        
        
        self.conn = conn
        self.vector_db_bank = vector_db_bank
        self.vector_db_non_bank = vector_db_non_bank
        self.vector_db_securities = vector_db_securities
        self.vector_db_ratio = vector_db_ratio
        
        self.vector_db_company = vector_db_company
        self.vector_db_sql = vector_db_sql
        self.multi_thread = multi_thread
    
    
    # Search for columns in bank and non_bank financial report
    def search(self, texts, top_k, type_) -> list:
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
    
    
    def search_multithread(self, texts, top_k, type_):
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
    
    # Execute SQL query
    def query(self, query, return_type='dataframe'):
        return execute_query(query, self.conn, return_type)
    
    def find_stock_code_similarity(self, company_name, top_k=2):
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
    
    def find_stock_code_similarity_multithread(self, company_name, top_k=2):
        start = time.time()
        
        # Override the original multi_thread
        original_multi_thread = self.multi_thread
        self.multi_thread = True
        
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

        # Reset the multi_thread
        self.multi_thread = original_multi_thread
        
        end = time.time()
        logging.info(f"Time taken to find stock code similarity multithread: {end-start}")
        return list(stock_codes)
    
    
    
    def return_company_from_stock_codes(self, stock_codes):
        if not isinstance(stock_codes, list):
            stock_codes = [stock_codes]
        stock_codes = [f"'{code}'" for code in stock_codes]
        
        # If no stock code found
        if len(stock_codes) == 0:
            return pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        
        query = f"SELECT stock_code, company_name, en_company_name, industry, is_bank, is_securities FROM company_info WHERE stock_code IN ({', '.join(stock_codes)})"
        return self.query(query, return_type='dataframe')
    
    
    
    def return_company_info(self, company_name, top_k=2):
        
        if self.multi_thread:
            stock_codes = self.find_stock_code_similarity_multithread(company_name, top_k)
        else:
            stock_codes = self.find_stock_code_similarity(company_name, top_k)
        
        df = self.return_company_from_stock_codes(stock_codes)
        if isinstance(df, str):
            return pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        
        df.drop_duplicates(subset=['stock_code'], inplace=True)
        return df

    # Find SQL query for few shot learning
    def find_sql_query(self, text, top_k=1):
        results = self.vector_db_sql.similarity_search(text, top_k)
        
        few_shot = ""
        for result in results:
            if result.metadata.get('sql_code', None) is not None:
                few_shot += '#### '+result.page_content + '\n\n'
                few_shot += f"```sql\n\n{result.metadata['sql_code']}```"
                
        return few_shot
    
    
    def __get_exact_industry_bm25(self, industries):
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
            
    
    def return_mapping_table_v1(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
        
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
    
    def return_mapping_table_v2(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = True):
        
        if not self.multi_thread:
            logging.info("Multi-thread is disabled. Using single thread (v1)")
            return self.return_mapping_table_v1(financial_statement_row, financial_ratio_row, industry, stock_code, top_k, get_all_tables)
        
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
    
    
def setup_db(model_name = 'text-embedding-3-small', multi_thread = True):
    conn = {
        'db_name': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    print(conn['db_name'])
    
    conn = connect_to_db(**conn)
    current_directory = os.path.dirname(__file__)
    
    collection_chromadb = 'category_bank_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_bank_chroma2')
    bank_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'category_non_bank_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_non_bank_chroma2')
    none_bank_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'category_sec_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_sec_chroma2')
    sec_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'category_ratio_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_ratio_chroma2')
    ratio_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'company_name_chroma'
    persist_directory = os.path.join(current_directory, '../data/company_name_chroma2')
    vector_db_company = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'sql_query'
    persist_directory = os.path.join(current_directory, '../data/sql_query2')
    vector_db_sql = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    return DBHUB(conn, bank_vector_store, none_bank_vector_store, sec_vector_store, ratio_vector_store, vector_db_company, vector_db_sql, multi_thread)
    
    
def setup_db_openai(model_name = 'text-embedding-3-small', multi_thread = True):
    conn = {
        'db_name': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    print(conn['db_name'])
    
    conn = connect_to_db(**conn)
    current_directory = os.path.dirname(__file__)
    
    collection_chromadb = 'category_bank_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_bank_chroma')
    bank_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'category_non_bank_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_non_bank_chroma')
    none_bank_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'category_sec_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_sec_chroma')
    sec_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'category_ratio_chroma'
    persist_directory = os.path.join(current_directory, '../data/category_ratio_chroma')
    ratio_vector_store = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'company_name_chroma'
    persist_directory = os.path.join(current_directory, '../data/company_name_chroma')
    vector_db_company = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    collection_chromadb = 'sql_query'
    persist_directory = os.path.join(current_directory, '../data/sql_query')
    vector_db_sql = create_chroma_db(collection_chromadb, persist_directory, model_name)
    
    return DBHUB(conn, bank_vector_store, none_bank_vector_store, sec_vector_store, ratio_vector_store, vector_db_company, vector_db_sql, multi_thread)

    
if __name__ == "__main__":
    model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': 'cuda'})
    db = setup_db(model, multi_thread=False)
    # db = setup_db()
    # print(db.search_return_df('total assets', 2, 'bank'))
    financial_row = ['total assets', 'total liabilities', 'total equity', 'net income']
    ratio_row = ['return on assets', 'return on equity', 'net profit margin', ]
    industry = ['Banking', 'Financial Services']
    stock_code = [ 'BID', 'VCB', 'VND']
    company_name = ['Vietcombank']
    
    # Trigger
    # db.__get_exact_industry_bm25("Financial Services")
    db.search_return_df('total assets', 2, 'bank')
    # db.return_mapping_table_v1(financial_row, ratio_row, industry, stock_code, top_k=5, get_all_tables=False)
    # db.return_mapping_table_v2(financial_row, ratio_row, industry, stock_code, top_k=5, get_all_tables=False)
    
    db.find_stock_code_similarity(company_name, top_k=2)
    db.find_stock_code_similarity_multithread(company_name, top_k=2)