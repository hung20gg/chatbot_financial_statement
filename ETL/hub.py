from langchain_chroma import Chroma
import re

from .connector import *
class DBHUB:
    """
    This will be the hub for both similarity search and rational DB
    """

    def __init__(self, conn, 
                 vector_db_bank: Chroma, 
                 vector_db_non_bank: Chroma, 
                 vector_db_securities: Chroma, 
                 vector_db_ratio: Chroma, 
                 vector_industry: Chroma,
                 vector_db_company: Chroma, 
                 vector_db_sql: Chroma):
        
        
        self.conn = conn
        self.vector_db_bank = vector_db_bank
        self.vector_db_non_bank = vector_db_non_bank
        self.vector_db_securities = vector_db_securities
        self.vector_db_ratio = vector_db_ratio
        
        self.vector_industry = vector_industry
        self.vector_db_company = vector_db_company
        self.vector_db_sql = vector_db_sql

    
    
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
                collect_code.add(item.metadata['code'])
        return list(collect_code)
    
    def search_return_df(self, text, top_k, type_ = 'non_bank') -> pd.DataFrame:
        collect_code = self.search(text, top_k, type_)
        collect_code = [f"'{code}'" for code in collect_code]
        
        query = ""

        query = f"SELECT category_code, en_caption FROM map_category_code_{type_} WHERE category_code IN ({', '.join(collect_code)})"
        return self.query(query,return_type='dataframe')
    
    # Execute SQL query
    def query(self, query, return_type='dataframe'):
        return execute_query(query, self.conn, return_type)
    
    def find_stock_code_similarity(self, company_name, top_k=2):
        if isinstance(company_name, str):
            company_name = [company_name]
        stock_codes = set()
        for name in company_name:
            result = self.vector_db_company.similarity_search(name, top_k)
            for item in result:
                stock_codes.add(item.metadata['stock_code'])
            
        return list(stock_codes)
    
    
    def return_company_from_stock_codes(self, stock_codes):
        if not isinstance(stock_codes, list):
            stock_codes = [stock_codes]
        stock_codes = [f"'{code}'" for code in stock_codes]
        
        # If no stock code found
        if len(stock_codes) == 0:
            return pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_security'])
        
        query = f"SELECT stock_code, company_name, en_company_name, industry, is_bank, is_security FROM company_info WHERE stock_code IN ({', '.join(stock_codes)})"
        return self.query(query, return_type='dataframe')
    
    
    def return_company_info(self, company_name, top_k=2):
        stock_codes = self.find_stock_code_similarity(company_name, top_k)
        
        df = self.return_company_from_stock_codes(stock_codes)
        df.drop_duplicates(subset=['stock_code'], inplace=True)
        return df

    # Find SQL query for few shot learning
    def find_sql_query(self, text, top_k=1):
        result = self.vector_db_sql.similarity_search(text, top_k)
        if result[0].metadata.get('sql_code', None) is not None:
            return f"```sql\n\n{result[0].metadata['sql_code']}```"
        return 'No SQL query found'
    
    def return_mapping_table(self, bank_column = [], non_bank_column = [], top_k = 5):
        bank_column_table = ""
        bank_exact_column = []
        for col in bank_column:
            result = self.vector_db_bank.similarity_search(col, top_k)
            for item in result:
                bank_exact_column.append(item.metadata['code'])
        
        # Get into PostgreSQL to get the exact column name. Should be moved to the Chroma DB
        if len(bank_exact_column) > 0:
            bank_exact_column = [f"'{code}'" for code in bank_exact_column]
            query = f"SELECT category_code, en_caption FROM map_category_code_bank WHERE category_code IN ({', '.join(bank_exact_column)})"
            bank_column_table = self.query(query, return_type='dataframe')
                
        non_bank_column_table = ""
        non_bank_exact_column = []
        for col in non_bank_column:
            result = self.vector_db_non_bank.similarity_search(col, top_k)
            for item in result:
                non_bank_exact_column.append(item.metadata['code'])
        
        if len(non_bank_exact_column) > 0:
            non_bank_exact_column = [f"'{code}'" for code in non_bank_exact_column]
            query = f"SELECT category_code, en_caption FROM map_category_code_non_bank WHERE category_code IN ({', '.join(non_bank_exact_column)})"
            non_bank_column_table = self.query(query, return_type='dataframe')   
        
        return bank_column_table, non_bank_column_table
    
    def return_mapping_table_v1(self, 
                                bank_column=[], 
                                non_bank_column=[], 
                                sec_bank_column=[], 
                                financial_ratio_row=[], top_k=5):
        
        raise NotImplementedError("This function is not implemented yet")
    
    
    def return_mapping_table_v2(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5):
        
        check_status_table = {
            'map_category_code_non_bank': True,
            'map_category_code_bank': True,
            'map_category_code_securities': True,
            'map_category_code_ratio': True
        }
        
        if len(stock_code) != 0:
            company_df = self.return_company_from_stock_codes(stock_code)
            
            if company_df['is_bank'].sum() == 0:
                check_status_table['map_category_code_bank'] = False
            if company_df['is_security'].sum() == 0:
                check_status_table['map_category_code_securities'] = False
            if company_df['is_bank'].sum() + company_df['is_security'].sum() == len(company_df):
                check_status_table['map_category_code_non_bank'] = False     
         
        # Avoid override from the previous check
        if len(industry) != 0:
            for ind in industry:
                result = self.vector_industry.similarity_search(ind, 1)
                if result[0].metadata['code'] == 'Banking':
                    check_status_table['map_category_code_non_bank'] = True
                if result[0].metadata['code'] == 'Financial Services':
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
                return_table['map_category_code_non_bank'] = self.search_return_df(financial_statement_row, top_k, is_bank=False)
            if check_status_table['map_category_code_bank']:
                return_table['map_category_code_bank'] = self.search_return_df(financial_statement_row, top_k, is_bank=True)
            if check_status_table['map_category_code_securities']:
                return_table['map_category_code_securities'] = self.search_return_df(financial_statement_row, top_k, is_security=True)
                
        if len(financial_ratio_row) != 0:
            return_table['map_category_code_ratio'] = self.search_return_df(financial_ratio_row, top_k, type_='ratio')
            
        return return_table