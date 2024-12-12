import os
import dotenv
dotenv.load_dotenv()

import psycopg2
import pandas as pd
import re

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface  import (
    HuggingFaceEmbeddings,
)

import logging
import time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#=================#
#       RDB       #
#=================#

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
        logging.info(f'Table {table_name} created successfully.')
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
        
        
def load_csv_to_postgres(table_name, csv_path, primary_key=None, foreign_key: dict = {}, **db_conn):
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Connect to the PostgreSQL database
    conn = connect_to_db(**db_conn)
    
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
        
        
        
def execute_query(query, conn=None, params = None, return_type='dataframe'):
    if conn is None:
        raise ValueError("Connection is not provided")
    
    close = False
    if isinstance(conn, dict):
        close = True
        conn = connect_to_db(**conn)
    try:
        with conn.cursor() as cur:
            
            cur.execute(query, params)
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


#=================#
#    Vector DB    #
#=================#

def create_chroma_db(collection_name, persist_directory, model_name='text-embedding-3-small'):
    if isinstance(model_name, str):
        if 'text-embedding' in model_name:
            embedding_function = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
        else:
            embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    else:
        embedding_function = model_name
    
    return Chroma(collection_name=collection_name, 
                  embedding_function=embedding_function, 
                  persist_directory=persist_directory)


#==================#
#  Setup VectorDB  #
#==================#


def setup_chroma_db_fs(collection_name, persist_directory, table, model_name='text-embedding-3-small', **db_conn):
    conn = connect_to_db(**db_conn)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT vi_caption, en_caption, category_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1], category[2]) for category in categories]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory, model_name)
    
    for category in categories:
        print(category)
        chroma_db.add_texts([category[0]], metadatas=[{'lang': 'vi', 'code': category[2]}])
        chroma_db.add_texts([category[1]], metadatas=[{'lang': 'en', 'code': category[2]}])
        
        
def setup_chroma_db_ratio(collection_name, persist_directory, table, model_name='text-embedding-3-small', **db_conn):
    conn = connect_to_db(**db_conn)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT ratio_name, ratio_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1]) for category in categories]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory, model_name)
    
    for category in categories:
        print(category)
        chroma_db.add_texts([category[0]], metadatas=[{'lang': 'en', 'code': category[1]}])
        
def setup_chroma_db_company_name(collection_name, persist_directory, table, model_name='text-embedding-3-small', **db_conn):
    conn = connect_to_db(**db_conn)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT  stock_code, company_name, en_company_name, en_short_name  FROM {table}")
            companies = cur.fetchall()
            companies = [(company[0], company[1], company[2], company[3]) for company in companies]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory, model_name)
    
    for company in companies:
        print(company)
        chroma_db.add_texts([company[0]], metadatas=[{'lang': 'vi', 'stock_code': company[0]}])
        chroma_db.add_texts([company[1]], metadatas=[{'lang': 'vi', 'stock_code': company[0]}])
        chroma_db.add_texts([company[2]], metadatas=[{'lang': 'en', 'stock_code': company[0]}])
        chroma_db.add_texts([company[3]], metadatas=[{'lang': 'en', 'stock_code': company[0]}])
        
def setup_chroma_db_sql_query(collection_name, persist_directory, txt_path, model_name='text-embedding-3-small'):
    with open(txt_path, 'r') as f:
        content = f.read()
    chroma_db = create_chroma_db(collection_name, persist_directory, model_name)
    sql = re.split(r'--\s*\d+', content)
    heading = re.findall(r'--\s*\d+', content)
    codes = []
    for i, s in enumerate(sql[1:]):
        sql_code = heading[i]+ s
        task = sql_code.split('\n')[0]
        task = re.sub(r'--\s*\d+\.?', '', task).strip()
        print(task)
        print(sql_code)
        print('====================')
        
                
    for code in codes:
        chroma_db.add_texts([code[0]], metadatas=[{'lang': 'sql', 'sql_code': code[1]}])

#================#
#  Setup config  #
#================#

RDB_SETUP_CONFIG = {
    'company_info' : ['../csv/df_company_info.csv', ['stock_code']],
    'map_category_code_bank': ['../csv/map_category_code_bank.csv', ['category_code']],
    'map_category_code_non_bank': ['../csv/map_category_code_non_bank.csv', ['category_code']],
    'map_category_code_securities': ['../csv/map_category_code_sec.csv', ['category_code']],
    'map_category_code_ratio': ['../csv/map_ratio_code.csv', ['ratio_code']],
    'sub_and_shareholder': ['../csv/df_sub_and_shareholder.csv', None, {'stock_code': 'company_info(stock_code)'}],
    
    'bank_financial_report' : ['../csv/bank_financial_report_v2_1.csv', None, {'category_code': 'map_category_code_bank(category_code)', 'stock_code': 'company_info(stock_code)'}],
    'non_bank_financial_report' : ['../csv/non_bank_financial_report_v2_1.csv', None, {'category_code': 'map_category_code_non_bank(category_code)', 'stock_code': 'company_info(stock_code)'}],
    'securities_financial_report' : ['../csv/securities_financial_report_v2_1.csv', None, {'category_code': 'map_category_code_securities(category_code)', 'stock_code': 'company_info(stock_code)'}],
    'financial_ratio' : ['../csv/financial_ratio.csv', None, {'ratio_code': 'map_category_code_ratio(ratio_code)', 'stock_code': 'company_info(stock_code)'}],
}

LOCAL_VERTICAL_BASE_VECTORDB_SETUP_CONFIG = {
    'company_name_chroma': ['../data/company_name_chroma_local', 'company_info'],
    'category_bank_chroma': ['../data/category_bank_chroma_local', 'map_category_code_bank'],
    'category_non_bank_chroma': ['../data/category_non_bank_chroma_local', 'map_category_code_non_bank'],
    'category_sec_chroma': ['../data/category_sec_chroma_local', 'map_category_code_securities'],
    'category_ratio_chroma': ['../data/category_ratio_chroma_local', 'map_category_code_ratio'],
    'sql_query': ['../data/sql_query_local', '../agent/prompt/vertical/base/simple_query_v2.txt'],
}

OPENAI_VERTICAL_BASE_VECTORDB_SETUP_CONFIG = {
    'company_name_chroma': ['../data/company_name_chroma_openai', 'company_info'],
    'category_bank_chroma': ['../data/category_bank_chroma_openai', 'map_category_code_bank'],
    'category_non_bank_chroma': ['../data/category_non_bank_chroma_openai', 'map_category_code_non_bank'],
    'category_sec_chroma': ['../data/category_sec_chroma_openai', 'map_category_code_securities'],
    'category_ratio_chroma': ['../data/category_ratio_chroma_openai', 'map_category_code_ratio'],
    'sql_query': ['../data/sql_query_openai', '../agent/prompt/vertical/base/simple_query_v2.txt'],
}

def setup_rdb(config, **db_conn):
    for table, params in config.items():
        args = [table] + params
        load_csv_to_postgres(*args, **db_conn)
        
def setup_vector_db(config, model_name = 'text-embedding-3-small', **db_conn):
    for table, params in config.items():
        params.append(model_name)
        if table == 'sql_query':
            setup_chroma_db_sql_query(table, *params)
        elif table == 'company_name_chroma':
            setup_chroma_db_company_name(table, *params, **db_conn)
        elif table == 'category_ratio_chroma':
            setup_chroma_db_ratio(table, *params, **db_conn)
        else:
            setup_chroma_db_fs(table, *params, **db_conn)
            
            
def main():
    db_conn = {
        'db_name': 'test_db',
        'user': 'postgres',
        'password': '12345678',
        'host': 'localhost',
        'port': '5433'
        
    }
    
    model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': 'cuda'})
    
    # setup_rdb(RDB_SETUP_CONFIG, **db_conn)
    # logging.info("RDB setup completed")
    # setup_vector_db(VECTOR_DB_SETUP_CONFIG, **db_conn)
    setup_vector_db(LOCAL_VERTICAL_BASE_VECTORDB_SETUP_CONFIG, model, **db_conn)
    logging.info("Vector DB setup completed")
            
if __name__ == '__main__':
    main()



    