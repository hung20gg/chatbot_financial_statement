import psycopg2
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import dotenv
import re

dotenv.load_dotenv()

industry_trans = {
    "Ngân hàng": "Banking",
    "Bất động sản": "Real Estate",
    "Thực phẩm và đồ uống": "Food and Beverage",
    "Điện, nước & xăng dầu khí đốt": "Electricity, Water & Gasoline/Oil and Gas",
    "Bảo hiểm": "Insurance",
    "Công nghệ Thông tin": "Information Technology",
    "Hóa chất": "Chemicals",
    "Tài nguyên Cơ bản": "Basic Resources",
    "Bán lẻ": "Retail",
    "Dầu khí": "Oil and Gas",
    "Dịch vụ tài chính": "Financial Services",
    "Du lịch và Giải trí": "Tourism and Entertainment"
}

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
def create_chroma_db(collection_name, persist_directory):
    embedding_function = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
    
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
        
    for code in codes:
        chroma_db.add_texts([code[0]], metadatas=[{'lang': 'sql', 'sql_code': code[1]}])

        
class DBHUB:
    """
    This will be the hub for both similarity search and rational DB
    """

    def __init__(self, conn, vector_db_bank: Chroma, vector_db_non_bank: Chroma, vector_db_company: Chroma, vector_db_sql: Chroma):
        self.conn = conn
        self.vector_db_bank = vector_db_bank
        self.vector_db_non_bank = vector_db_non_bank
        self.vector_db_company = vector_db_company
        self.vector_db_sql = vector_db_sql

    
    
    # Search for columns in bank and non_bank financial report
    def search(self, texts, top_k, is_bank) -> list:
        collect_code = set()
        if not isinstance(texts, list):
            texts = [texts]
        for text in texts:
            if is_bank:
                result = self.vector_db_bank.similarity_search(text, top_k)
            else:
                result = self.vector_db_non_bank.similarity_search(text, top_k)
            
            for item in result:
                collect_code.add(item.metadata['code'])
        return list(collect_code)
    
    def search_return_df(self, text, top_k, is_bank = False) -> pd.DataFrame:
        collect_code = self.search(text, top_k, is_bank)
        collect_code = [f"'{code}'" for code in collect_code]
        query = f"SELECT category_code, en_caption FROM map_category_code_{'' if is_bank else 'non_'}bank WHERE category_code IN ({', '.join(collect_code)})"
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
    
    def return_company_info(self, company_name, top_k=2):
        stock_codes = self.find_stock_code_similarity(company_name, top_k)
        stock_codes = [f"'{code}'" for code in stock_codes]
        query = f"SELECT stock_code, company_name, en_company_name, industry, is_bank FROM company_info WHERE stock_code IN ({', '.join(stock_codes)})"
        return self.query(query, return_type='dataframe')

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

# Example usage
if __name__ == '__main__':
    # Database connection details
    db_name = 'test_db'
    user = 'postgres'
    password = '12345678'
    port = '5433'
    host = 'localhost'
    
    csv_path_company_info = 'sample_data/df_company_info.csv'
    table_name_company_info = 'company_info'

    # Primary and foreign key definitions
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
    
    # csv_path = 'sample_data/map_category_code_non_bank.csv'
    # table_name = 'map_category_code_non_bank'
    # collection_chromadb = 'category_non_bank_chroma'
    # persist_directory = 'data/category_non_bank_chroma'

    # # Load csv data to PostgreSQL

    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    # print("Loaded map_category_code_non_bank")
    # # setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    # print("Setup Chroma DB for map_category_code_non_bank")
    # # Generate embeddings  for the data
    
    
    # csv_path = 'sample_data/map_category_code_bank.csv'
    # table_name = 'map_category_code_bank'
    # collection_chromadb = 'category_bank_chroma'
    # persist_directory = 'data/category_bank_chroma'

    # # Load csv data to PostgreSQL
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    # print("Loaded map_category_code_bank")
    # # setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    # print("Setup Chroma DB for map_category_code_bank")
    
    
    # # Load Bank Financial Report
    # csv_path = 'sample_data/bank_financial_report.csv'
    # table_name = 'bank_financial_report'
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_bank(category_code)', 'stock_code': 'company_info(stock_code)'})
    # print("Loaded bank_financial_report")
    
    # # # Load Non Bank Financial Report
    # csv_path = 'sample_data/non_bank_financial_report.csv'
    # table_name = 'non_bank_financial_report'
    # load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_non_bank(category_code)', 'stock_code': 'company_info(stock_code)'})
    # print("Loaded non_bank_financial_report")

    # # Paths for the uploaded CSV files
    
    
    # # Setup Chroma DB for company_info
    # collection_chromadb = 'company_name_chroma'
    # persist_directory = 'data/company_name_chroma'
    # # setup_chroma_db_company_name(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name_company_info)

    collection_chromadb = 'sql_query'
    persist_directory = 'data/sql_query'
    setup_chroma_db_sql_query(collection_chromadb, persist_directory, 'prompt/question_query.txt')
    setup_chroma_db_sql_query(collection_chromadb, persist_directory, 'prompt/simple_query_v2.txt')

    # # Load 'sub_and_shareholder' data into PostgreSQL with foreign key relationship
    # csv_path_sub_and_shareholder = 'sample_data/df_sub_and_shareholder.csv'
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

