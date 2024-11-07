import psycopg2
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import dotenv
import re

dotenv.load_dotenv()

# Step 1: Connect to PostgreSQL
def connect_to_db(db_name, user, password, host='localhost', port='5432'):
    print(f'Connecting to database {db_name}, {user}...')
    
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn

# Step 2: Create table if it doesn't exist
def create_table_if_not_exists(conn, table_name, df, primary_key=None, foreign_key: dict = {},long_text=True):
    columns = df.columns
    col_type = []
    
    if primary_key is None:
        primary_key = set()
    else:
        primary_key = set(primary_key)
    
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            max_num = df[col].max()
            if max_num > 100_000_000:
                col_type.append('DECIMAL')
            else:
                col_type.append('INTEGER')
        elif pd.api.types.is_float_dtype(df[col]):
            col_type.append('FLOAT')
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type.append('TIMESTAMP')
        elif pd.api.types.is_bool_dtype(df[col]):
            col_type.append('BOOLEAN')
        else:
            df[col] = df[col].astype(str)
            max_len = df[col].str.len().max()
            if long_text and max_len > 255:
                col_type.append('TEXT')
            else:
                col_type.append('VARCHAR(255)')

    with conn.cursor() as cur:
        # Replace this with the appropriate table creation logic based on your CSV structure
        column_definitions = ""
        
        for col, type_ in zip(columns, col_type):
            column_definitions += f'{col} {type_} '
            if col in primary_key:
                column_definitions += 'PRIMARY KEY '
            if foreign_key.get(col):
                column_definitions += f'REFERENCES {foreign_key[col]} '
                
            column_definitions += ', '
        
        column_definitions = column_definitions[:-2]
        cur.execute(f"""
            DROP TABLE IF EXISTS {table_name};        
                    
            CREATE TABLE {table_name} (
                {column_definitions}
            );
        """)
        print(f'Table {table_name} created successfully.')
        conn.commit()

# Step 3: Insert data into table (upsert logic)
def upsert_data(conn, table_name, df, log_gap = 1000):
    with conn.cursor() as cur:
        # Define a placeholder for the insert values
        placeholders = ', '.join(['%s'] * len(df.columns))
        # Convert DataFrame to list of tuples
        data_tuples = [tuple(x) for x in df.to_numpy()]
        
        # Perform the upsert operation
        for i,row in enumerate(data_tuples):
            upsert_query = f"""
                INSERT INTO {table_name} VALUES ({placeholders})
            """
            cur.execute(upsert_query, row)
            if i%log_gap == 0:
                print(f'Upserted row: {row}')
        
        conn.commit()

# Step 4: Load CSV and call the functions
def load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=None, foreign_key: dict = {},long_text=False):
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Connect to the PostgreSQL database
    conn = connect_to_db(db_name, user, password, port=port)
    
    try:
        # Create the table if it doesn't exist
        print('Creating table in database...')
        create_table_if_not_exists(conn, table_name, df, primary_key, foreign_key)
        
        # Upsert the data into the table
        print('Upserting data into the table...')
        upsert_data(conn, table_name, df)
    finally:
        print('Closing connection to database...')
        conn.close()

# Step 5: Execute SQL Query
def execute_query(query, conn=None, return_type='tuple'):
    if conn is None:
        raise ValueError("Connection is not provided")
    
    close = False
    if isinstance(conn, dict):
        close = True
        conn = connect_to_db(**conn)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            
            if return_type == 'dataframe':
                columns = [desc[0] for desc in cur.description]
                result = pd.DataFrame(result, columns=columns)
    except Exception as e:
        print(e)
        result = str(e) 
    finally:
        if close:
            conn.close()
    return result

# Step 6: Create Chroma DB
def create_chroma_db(collection_name, persist_directory, model_name='text-embedding-3-small'):
    if 'text-embedding' in model_name:
        embedding_function = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
    else:
        raise ValueError("Model name not supported")
    
    return Chroma(collection_name=collection_name, 
                  embedding_function=embedding_function, 
                  persist_directory=persist_directory)

# Step 7: Setup Chroma DB
def setup_chroma_db_fs(db_name, user, password, host, port, collection_name, persist_directory, table='map_category_non_bank'):
    conn = connect_to_db(db_name, user, password, host, port)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT vi_caption, en_caption, category_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1], category[2]) for category in categories]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory)
    
    for category in categories:
        print(category)
        chroma_db.add_texts([category[0]], metadatas=[{'lang': 'vi', 'code': category[2]}])
        chroma_db.add_texts([category[1]], metadatas=[{'lang': 'en', 'code': category[2]}])
  
def setup_chroma_db_ratio(db_name, user, password, host, port, collection_name, persist_directory, table='map_category_non_sec'):
    conn = connect_to_db(db_name, user, password, host, port)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT ratio_name, ratio_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1]) for category in categories]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory)
    
    for category in categories:
        print(category)
        chroma_db.add_texts([category[0]], metadatas=[{'lang': 'en', 'code': category[1]}])
        
def setup_chroma_db_company_name(db_name, user, password, host, port, collection_name, persist_directory, table='company_info'):
    conn = connect_to_db(db_name, user, password, host, port)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT  stock_code, company_name, en_company_name, en_short_name  FROM {table}")
            companies = cur.fetchall()
            companies = [(company[0], company[1], company[2], company[3]) for company in companies]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory)
    
    for company in companies:
        print(company)
        chroma_db.add_texts([company[0]], metadatas=[{'lang': 'vi', 'stock_code': company[0]}])
        chroma_db.add_texts([company[1]], metadatas=[{'lang': 'vi', 'stock_code': company[0]}])
        chroma_db.add_texts([company[2]], metadatas=[{'lang': 'en', 'stock_code': company[0]}])
        chroma_db.add_texts([company[3]], metadatas=[{'lang': 'en', 'stock_code': company[0]}])
        
def setup_chroma_db_sql_query(collection_name, persist_directory, txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
    chroma_db = create_chroma_db(collection_name, persist_directory)
    sql = re.split(r'--\s*\d+', content)
    heading = re.findall(r'--\s*\d+', content)
    codes = []
    for i, s in enumerate(sql[1:]):
        sql_code = heading[i]+ s
        task = sql_code.split('\n')[0]
        task = re.sub(r'--\s*\d+\.?', '', task).strip()
        codes.append((task, sql_code))
        print(task)
        print(sql_code)
        print('====================')
        
                
    for code in codes:
        chroma_db.add_texts([code[0]], metadatas=[{'lang': 'sql', 'sql_code': code[1]}])

        
class DBHUB:
    """
    This will be the hub for both similarity search and rational DB
    """

    def __init__(self, conn, 
                 vector_db_bank: Chroma, 
                 vector_db_non_bank: Chroma, 
                 vector_db_securities: Chroma, 
                 vector_db_ratio: Chroma, 
                 vector_db_company: Chroma, 
                 vector_db_sql: Chroma):
        
        
        self.conn = conn
        self.vector_db_bank = vector_db_bank
        self.vector_db_non_bank = vector_db_non_bank
        self.vector_db_securities = vector_db_securities
        self.vector_db_ratio = vector_db_ratio
        
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
        if type_ == 'ratio':
            query = f"SELECT ratio_code, ratio_name FROM map_category_code_ratio WHERE ratio_code IN ({', '.join(collect_code)})"
        else:
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
            return pd.DataFrame(columns=['stock_code', 'company_name', 'en_company_name', 'industry', 'is_bank', 'is_securities'])
        
        query = f"SELECT stock_code, company_name, en_company_name, industry, is_bank, is_securities FROM company_info WHERE stock_code IN ({', '.join(stock_codes)})"
        return self.query(query, return_type='dataframe')
    
    
    def return_company_info(self, company_name, top_k=2):
        stock_codes = self.find_stock_code_similarity(company_name, top_k)
        
        df = self.return_company_from_stock_codes(stock_codes)
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
                                financial_ratio_row=[], top_k=5, *args, **kwargs):
        
        raise NotImplementedError("This function is not implemented yet")
    
    def __get_exact_industry_bm25(self, industries):
        query = """
        SELECT distinct (industry)
FROM company_info
WHERE industry_tsvector @@ to_tsquery('english', '{industry}');
        """
        if not isinstance(industries, list):
            industries = [industries]
        exact_industries = set()
        for industry in industries:
            df = self.query(query.format(industry=industry))
            result = df['industry'].values.tolist()
            for item in result:
                exact_industries.add(item[0])
        return list(exact_industries)
            
    
    def return_mapping_table_v2(self, financial_statement_row = [], financial_ratio_row = [], industry = [], stock_code = [], top_k =5, get_all_tables = False):
        
        check_status_table = {
            'map_category_code_non_bank': True,
            'map_category_code_bank': True,
            'map_category_code_securities': True,
            'map_category_code_ratio': True
        }
        
        if len(stock_code) != 0 and not get_all_tables:
            company_df = self.return_company_from_stock_codes(stock_code)
            
            if company_df['is_bank'].sum() == 0:
                check_status_table['map_category_code_bank'] = False
            if company_df['is_security'].sum() == 0:
                check_status_table['map_category_code_securities'] = False
            if company_df['is_bank'].sum() + company_df['is_security'].sum() == len(company_df):
                check_status_table['map_category_code_non_bank'] = False     
         
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
            
        return return_table


# Example usage
if __name__ == '__main__':
    # Database connection details
    db_name = 'test_db'
    user = 'postgres'
    password = '12345678'
    port = '5433'
    host = 'localhost'
    
    # It is recommended to delete all the existing db, vector db and load the data in the following order:
    
    # # Load general data into df
    
    # csv_path_company_info = '../csv/df_company_info.csv'
    # table_name_company_info = 'company_info'

    # # Primary and foreign key definitions
    # primary_key_company_info = ['stock_code']
    # primary_key_sub_and_shareholder = None

    # # Load 'company_info' data into PostgreSQL
    # load_csv_to_postgres(
    #     csv_path=csv_path_company_info,
    #     db_name=db_name,
    #     user=user,
    #     password=password,
    #     table_name=table_name_company_info,
    #     port=port,
    #     primary_key=primary_key_company_info
    # )
    # print("Loaded company_info table")
    
    # # Setup Chroma DB for company_info
    # collection_chromadb = 'company_name_chroma'
    # persist_directory = 'data/company_name_chroma'
    # setup_chroma_db_company_name(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name_company_info)
    
    # csv_path = '../csv/map_category_code_non_bank.csv'
    # table_name = 'map_category_code_non_bank'
    # collection_chromadb = 'category_non_bank_chroma'
    # persist_directory = 'data/category_non_bank_chroma'

    # # Load csv data to PostgreSQL

    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    # print("Loaded map_category_code_non_bank")
    # # setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    # print(f"Setup Chroma DB for {table_name}")
    # # Generate embeddings  for the data
    
    
    # csv_path = '../csv/map_category_code_bank.csv'
    # table_name = 'map_category_code_bank'
    # collection_chromadb = 'category_bank_chroma'
    # persist_directory = 'data/category_bank_chroma'

    # # Load csv data to PostgreSQL
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    # print("Loaded map_category_code_bank")
    # # setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    # print(f"Setup Chroma DB for {table_name}")
    
    
    # csv_path = '../csv/map_category_code_sec.csv'
    # table_name = 'map_category_code_securities'
    # collection_chromadb = 'category_sec_chroma'
    # persist_directory = 'data/category_sec_chroma'

    # # Load csv data to PostgreSQL
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    # print("Loaded map_category_code_bank")
    # setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    # print(f"Setup Chroma DB for {table_name}")
    
    # csv_path = '../csv/map_ratio_code.csv'
    # table_name = 'map_category_code_ratio'
    # collection_chromadb = 'category_ratio_chroma'
    # persist_directory = 'data/category_ratio_chroma'

    # # Load csv data to PostgreSQL
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['ratio_code'])
    # print("Loaded map_category_code_bank")
    # setup_chroma_db_ratio(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    # print(f"Setup Chroma DB for {table_name}")
    
    # # Load financial record data 
    
    
    # # Load Bank Financial Report
    # csv_path = '../csv/bank_financial_report_v2_1.csv'
    # table_name = 'bank_financial_report'
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_bank(category_code)', 'stock_code': 'company_info(stock_code)'})
    # print(f"Loaded {table_name}")
    
    # # # Load Non Bank Financial Report
    # csv_path = '../csv/non_bank_financial_report_v2_1.csv'
    # table_name = 'non_bank_financial_report'
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_non_bank(category_code)', 'stock_code': 'company_info(stock_code)'})
    # print(f"Loaded {table_name}")

    # # # Load Securities Financial Report
    # csv_path = '../csv/securities_financial_report_v2_1.csv'
    # table_name = 'securities_financial_report'
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_securities(category_code)', 'stock_code': 'company_info(stock_code)'})
    # print(f"Loaded {table_name}")

    # # Load Financial Ratio
    csv_path = '../csv/financial_ratio.csv'
    table_name = 'financial_ratio'
    load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'ratio_code': 'map_category_code_ratio(ratio_code)', 'stock_code': 'company_info(stock_code)'})
    # print(f"Loaded {table_name}")
    



    # SQL Prompt Few shot

    # collection_chromadb = 'sql_query'
    # persist_directory = 'data/sql_query'
    # # setup_chroma_db_sql_query(collection_chromadb, persist_directory, 'prompt/question_query.txt')
    # setup_chroma_db_sql_query(collection_chromadb, persist_directory, 'prompt/simple_query_v2.txt')

    # # # Load 'sub_and_shareholder' data into PostgreSQL with foreign key relationship
    # csv_path_sub_and_shareholder = '../csv/df_sub_and_shareholders.csv'
    # table_name_sub_and_shareholder = 'sub_and_shareholder'
    
    # load_csv_to_postgres(
    #     csv_path=csv_path_sub_and_shareholder,
    #     db_name=db_name,
    #     user=user,
    #     password=password,
    #     table_name=table_name_sub_and_shareholder,
    #     port=port,
    #     foreign_key={'stock_code': 'company_info(stock_code)'}
    # )
    # print("Loaded sub_and_shareholder table")

