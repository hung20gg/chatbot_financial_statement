import os

import sys 
sys.path.append('..')

import dotenv
dotenv.load_dotenv()

from chromadb import PersistentClient
from chromadb.config import Settings
import subprocess

import psycopg2
import pandas as pd
import re

from langchain_chroma import Chroma
from langchain_milvus import Milvus
from langchain_elasticsearch import ElasticsearchStore


from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor

from langchain_huggingface  import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings
)

import logging
import requests
import time
import copy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


current_dir = os.path.dirname(os.path.abspath(__file__))    
BATCH_SIZE = 32

#=================#
#       RDB       #
#=================#

def connect_to_db(db_name, user, password, host='localhost', port='5432'):
    logging.info(f'Connecting to database {db_name}, {user}...')
    
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn


def check_table_exists(connection, table_name):
    query = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = %s
    );
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (table_name,))
            result = cursor.fetchone()[0]
            return result
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False
    

def insert_row_if_not_exists(connection, table_name, row_data):
    # Generate the column names and placeholders dynamically
    columns = ", ".join(row_data.keys())
    placeholders = ", ".join(["%s"] * len(row_data))
    
    # Build the SQL query to check if the row exists
    check_query = f"""
    SELECT EXISTS (
        SELECT 1 FROM {table_name} 
        WHERE ({columns}) = ({placeholders})
    );
    """
    
    # Build the SQL query to insert the row
    insert_query = f"""
    INSERT INTO {table_name} ({columns}) 
    VALUES ({placeholders});
    """
    
    try:
        with connection.cursor() as cursor:
            # Check if the row exists
            cursor.execute(check_query, tuple(row_data.values))
            exists = cursor.fetchone()[0]
            
            if not exists:
                # Insert the row if it doesn't exist
                cursor.execute(insert_query, tuple(row_data.values))

    except Exception as e:
        print(f"Error: {e}")
        
# Step 3.1: Insert data into table savely
def upsert_data_save(conn, table_name, df, log_gap = 5000):
    for i, row in df.iterrows():
        insert_row_if_not_exists(conn, table_name, row)
        
        if i%log_gap == 0:
            logging.info(f'Upserted row: {row}')


def create_table_if_not_exists(conn, table_name, df_path, primary_key=None, foreign_key: dict = {}, long_text=False, date_time = []):
    
    df_path = os.path.join(current_dir, df_path)
    
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path)
    elif df_path.endswith('.xlsx'):
        df = pd.read_excel(df_path)
    elif df_path.endswith('.parquet'):
        df = pd.read_parquet(df_path)
    
    columns = df.columns
    col_type = []
    
    for col in date_time:
        if col in columns:
            df[col] = pd.to_datetime(df[col])
    
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
            
            # if col in allow_null:
            #     column_definitions += 'NULL '
                
            if foreign_key.get(col):
                column_definitions += f'REFERENCES {foreign_key[col]} '
                
            
                
            column_definitions += ', '
        
        column_definitions = column_definitions[:-2]
        cur.execute(f"""
            DROP TABLE IF EXISTS {table_name};        
                    
            CREATE TABLE IF NOT EXISTS {table_name} (
                {column_definitions}
            );
        """)
        logging.info(f'Table {table_name} created successfully.')
        conn.commit()

# Step 3: Insert data into table (upsert logic)
def upsert_data(conn, table_name, df, log_gap = 5000):
    
    # Define a placeholder for the insert values
    placeholders = ', '.join(['%s'] * len(df.columns))
    # Convert DataFrame to list of tuples
    data_tuples = [tuple(x) for x in df.to_numpy()]

    upsert_query = """
        INSERT INTO {table_name}
        VALUES ({placeholders})
        """
    
    for i in range(0, len(data_tuples), log_gap):
        try:
            with conn.cursor() as cur:
                query = upsert_query.format(table_name=table_name, placeholders=placeholders)
                cur.executemany(query, data_tuples[i:i+log_gap])
                conn.commit()
                logging.info(f'Upserted rows: {i} to {i+log_gap}')

        except Exception as e:
            conn.rollback()
            logging.error(f"Error during upsert: {e}")

        
def load_csv_to_postgres(force = False, *args, **db_conn):
    # Load CSV into pandas DataFrame
    path = os.path.join(current_dir, args[1])
    
    if args[1].endswith('.csv'):
        df = pd.read_csv(path)
    elif args[1].endswith('.xlsx'):
        df = pd.read_excel(path)
    elif args[1].endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise ValueError("File format not supported")    
    
    # Connect to the PostgreSQL database
    conn = connect_to_db(**db_conn)
    
    try:
        # Create the table if it doesn't exist
        if not check_table_exists(conn, args[0]):
            logging.info('Creating table in database...')
            create_table_if_not_exists(conn, *args)
            
            # Upsert the data into the table
            upsert_data(conn, args[0], df)
            
        elif force:
            logging.info('Table already exists in database. Inserting data...')
            upsert_data_save(conn, args[0], df)
        else:
            logging.info('Table already exists in database. Skipping data insertion...')
            
    finally:
        conn.close()
        
        
def delete_table(conn, table_name):
    
    conn = connect_to_db(**conn)
    
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {table_name}")
            conn.commit()
            logging.info(f'Table {table_name} deleted successfully.')
    except Exception as e:
        logging.error(f'Error deleting table {table_name}: {e}. Perhaps the table does not exist.')
    
    conn.close()


def delete_tables(conn, table_names):
    for table_name in table_names:
        delete_table(conn, table_name)
    
        
        
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

def create_embedding_function(model_name):
    if isinstance(model_name, str):
        if 'text-embedding' in model_name:
            embedding_function = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
        
        elif model_name[:4] == 'http':
            embedding_function = HuggingFaceEndpointEmbeddings(model=model_name)
        
        else:
            embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    else:
        embedding_function = model_name
        
    return embedding_function
    


def create_chroma_db(collection_name, persist_directory, embedding_function):        
    if isinstance(persist_directory, str):
        return Chroma(collection_name=collection_name, 
                    embedding_function=embedding_function, 
                    persist_directory=persist_directory)
    else:
        return Chroma(collection_name=collection_name, 
                    embedding_function=embedding_function, 
                    client=persist_directory)


def create_milvus_db(collection_name, persist_directory, embedding_function):
    return Milvus(
        collection_name=collection_name,
        embedding_function=embedding_function,
        connection_args={"uri": persist_directory} # Either directory or an uri
    )
    
def create_elastic_search_db(collection_name, es_url, embedding_function):
    return ElasticsearchStore(
        index_name = collection_name,
        es_url = es_url,
        embedding=embedding_function
    )
    
def create_vector_db(collection_name, persist_directory, model_name, vectordb):
    
    embedding_function = create_embedding_function(model_name)
    
    if vectordb == 'milvus':
        return create_milvus_db(collection_name, persist_directory, embedding_function)
    elif vectordb == 'chromadb':
        return create_chroma_db(collection_name, persist_directory, embedding_function)
    elif vectordb == 'elastic-search':
        return create_elastic_search_db(collection_name, persist_directory, embedding_function)
    
    else:
        raise ValueError(f"Vectordb format {vectordb} is not supported")

#==================#
#  Setup VectorDB  #
#==================#


def setup_vector_db_fs(collection_name, persist_directory, table, model_name, vectordb, **db_conn):
    conn = connect_to_db(**db_conn)
    logging.info("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT vi_caption, en_caption, category_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1], category[2]) for category in categories]
    finally:
        conn.close()
    
    vector_db = create_vector_db(collection_name, persist_directory, model_name, vectordb)
    
    def process_category(vector_db, batch_category):
        
        categories_0 = [category[0] for category in batch_category]
        categories_1 = [category[1] for category in batch_category]
        
        metadatas_0 = [{'lang': 'vi', 'code': category[2]} for category in batch_category]
        metadatas_1 = [{'lang': 'en', 'code': category[2]} for category in batch_category]
        
        vector_db.add_texts(categories_0, metadatas=metadatas_0)
        vector_db.add_texts(categories_1, metadatas=metadatas_1)
        
    batch_categories = [categories[i:i+BATCH_SIZE] for i in range(0, len(categories), BATCH_SIZE)]
        
    with ThreadPoolExecutor() as executor:
        executor.map(lambda category: process_category(vector_db, category), batch_categories)
        

def setup_vector_db_universal(collection_name, persist_directory, table, model_name, vectordb, **db_conn):
    conn = connect_to_db(**db_conn)
    logging.info("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT en_caption, category_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1]) for category in categories]
    finally:
        conn.close()
        
    logging.info(f'Categories: {len(categories)}')
    
    vector_db = create_vector_db(collection_name, persist_directory, model_name, vectordb)
    
    def process_category(vector_db, batch_category):
        
        categories_0 = [category[0] for category in batch_category]
        metadatas = [{'lang': 'en', 'code': category[1]} for category in batch_category]
        
        vector_db.add_texts(categories_0, metadatas=metadatas)
        # chroma_db.add_texts([category[1]], metadatas=[{'lang': 'en', 'code': category[2]}])
        
    batch_categories = [categories[i:i+BATCH_SIZE] for i in range(0, len(categories), BATCH_SIZE)]
        
    with ThreadPoolExecutor() as executor:
        executor.map(lambda category: process_category(vector_db, category), batch_categories)
        
        
def setup_vector_db_ratio(collection_name, persist_directory, table, model_name, vectordb, **db_conn):
    conn = connect_to_db(**db_conn)
    print(conn)
    logging.info("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT ratio_name, ratio_code FROM {table}")
            categories = cur.fetchall()
            categories = [(category[0], category[1]) for category in categories]
    finally:
        conn.close()
    
    vector_db = create_vector_db(collection_name, persist_directory, model_name, vectordb)
    
    def process_category(vector_db, batch_category):
        
        categories_0 = [category[0] for category in batch_category]
        categories_1 = [category[1] for category in batch_category]
        
        metadatas = [{'lang': 'en', 'code': category[1]} for category in batch_category]
        
        vector_db.add_texts(categories_0, metadatas=metadatas)
        vector_db.add_texts(categories_1, metadatas=metadatas)
    
    batch_categories = [categories[i:i+BATCH_SIZE] for i in range(0, len(categories), BATCH_SIZE)]
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda category: process_category(vector_db, category), batch_categories)
        

#==================#
#   Setup Utils    #
#==================#
    
        
def setup_vector_db_company_name(collection_name, persist_directory, table, model_name, vectordb, **db_conn):
    conn = connect_to_db(**db_conn)
    logging.info("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT  stock_code, company_name, en_company_name, en_short_name  FROM {table}")
            companies = cur.fetchall()
            companies = [(company[0], company[1], company[2], company[3]) for company in companies]
    finally:
        conn.close()
    
    vector_db = create_vector_db(collection_name, persist_directory, model_name, vectordb)
    
    def process_company(vector_db, company):
        vector_db.add_texts(list(company), metadatas=[{'lang': 'vi', 'stock_code': company[0]}] * 4)
        
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda company: process_company(vector_db, company), companies)

        
def setup_vector_db_sql_query(collection_name, persist_directory, txt_path, model_name, vectordb, **db_conn):
    
    txt_path = os.path.join(current_dir, txt_path)
    
    with open(txt_path, 'r') as f:
        content = f.read()
    vector_db = create_vector_db(collection_name, persist_directory, model_name, vectordb)
    
    sql = re.split(r'--\s*\d+', content)
    heading = re.findall(r'--\s*\d+', content)
    codes = []
    for i, s in enumerate(sql[1:]):
        sql_code = heading[i]+ s
        task = sql_code.split('\n')[0]
        task = re.sub(r'--\s*\d+\.?', '', task).strip()
        
        codes.append((task, sql_code))
        
                
    for code in codes:
        vector_db.add_texts([code[0]], metadatas=[{'lang': 'sql', 'sql_code': code[1]}])

#================#
#  Setup config  #
#================#


RDB_SETUP_CONFIG = {
    'company_info' : ['../csv/df_company_info.csv', ['stock_code'], {}, True],
    'sub_and_shareholder': ['../csv/df_sub_and_shareholders.csv', None, {'stock_code': 'company_info(stock_code)'}],
    'map_category_code_bank': ['../csv/map_category_code_bank.csv', ['category_code']],
    'map_category_code_non_bank': ['../csv/map_category_code_non_bank.csv', ['category_code']],
    'map_category_code_securities': ['../csv/map_category_code_sec.csv', ['category_code']],
    'map_category_code_ratio': ['../csv/map_ratio_code.csv', ['ratio_code']],
    'map_category_code_universal': ['../csv/map_category_code_universal.csv', ['category_code']],
    
    
    'bank_financial_report' : ['../csv/bank_financial_report_v2.parquet', None, {'category_code': 'map_category_code_bank(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'non_bank_financial_report' : ['../csv/non_bank_financial_report_v2.parquet', None, {'category_code': 'map_category_code_non_bank(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'securities_financial_report' : ['../csv/securities_financial_report_v2.parquet', None, {'category_code': 'map_category_code_securities(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'financial_statement': ['../csv/financial_statement.parquet', None, {'category_code': 'map_category_code_universal(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],

    'financial_ratio' : ['../data/financial_ratio.parquet', None, {'ratio_code': 'map_category_code_ratio(ratio_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'industry_financial_statement': ['../data/industry_report_v2.parquet', None, {'category_code': 'map_category_code_universal(category_code)'}, False, ['date_added']],
    'industry_financial_ratio': ['../data/industry_ratio_v2.parquet', None, {'ratio_code': 'map_category_code_ratio(ratio_code)'}, False, ['date_added']],    
}

FIIN_RDB_SETUP_CONFIG = {
    'company_info' : ['../csv/df_company_info.csv', ['stock_code'], {}, True],
    'sub_and_shareholder': ['../csv/df_sub_and_shareholders.csv', None, {'stock_code': 'company_info(stock_code)'}],
    'map_category_code_bank': ['../csv/map_category_bank_v3.csv', ['category_code']],
    'map_category_code_non_bank': ['../csv/map_category_corp_v3.csv', ['category_code']],
    'map_category_code_securities': ['../csv/map_category_sec_v3.csv', ['category_code']],
    'map_category_code_ratio': ['../csv/map_ratio_code.csv', ['ratio_code']],
    'map_category_code_universal': ['../csv/map_category_code_universal_v3.csv', ['category_code']],
    
    
    'bank_financial_report' : ['../csv/bank_financial_report_v3.parquet', None, {'category_code': 'map_category_code_bank(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'non_bank_financial_report' : ['../csv/non_bank_financial_report_v3.parquet', None, {'category_code': 'map_category_code_non_bank(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'securities_financial_report' : ['../csv/securities_financial_report_v3.parquet', None, {'category_code': 'map_category_code_securities(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'financial_statement': ['../csv/financial_statement_v3.parquet', None, {'category_code': 'map_category_code_universal(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    

    'map_category_code_explaination': ['../csv/map_category_code_explaination_v3.csv', ['category_code']],
    'financial_statement_explaination': ['../csv/financial_statement_explaination_v3.parquet', None, {'category_code': 'map_category_code_explaination(category_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    
    
    'financial_ratio' : ['../data/financial_ratio_v3.parquet', None, {'ratio_code': 'map_category_code_ratio(ratio_code)', 'stock_code': 'company_info(stock_code)'}, False, ['date_added']],
    'industry_financial_statement': ['../data/industry_report_v3.parquet', None, {'category_code': 'map_category_code_universal(category_code)'}, False, ['date_added']],
    'industry_financial_ratio': ['../data/industry_ratio_v3.parquet', None, {'ratio_code': 'map_category_code_ratio(ratio_code)'}, False, ['date_added']],    

}


DELETE_ORDER = list(RDB_SETUP_CONFIG.keys())[::-1] # delete in reverse order
FIIN_DELETE_ORDER = list(FIIN_RDB_SETUP_CONFIG.keys())[::-1] # delete in reverse order


VERTICAL_VECTORDB_SETUP_CONFIG = {
    'company_name_chroma': ['company_info'],
    'category_bank_chroma': ['map_category_code_bank'],
    'category_non_bank_chroma': ['map_category_code_non_bank'],
    'category_sec_chroma': ['map_category_code_securities'],
    'category_ratio_chroma': ['map_category_code_ratio'],
    'sql_query': ['../agent/prompt/vertical/base/simple_query_v2.txt'],
    'sql_query_universal': ['../agent/prompt/vertical/universal/simple_query_v2.txt'],
    'category_universal_chroma': ['map_category_code_universal'],
    'category_universal_chroma$': ['map_category_code_explaination'],
}

def setup_rdb(force, config, **db_conn):
    for table, params in config.items():
        args = [table] + params
        load_csv_to_postgres(force, *args, **db_conn)
        

def remove_dollar_sign(text):
    return text.replace('$', '')

        
def setup_vector_db(config, persist_directory, model_name = 'text-embedding-3-small', vectordb = 'chromadb', **db_conn):
    
    config = copy.deepcopy(config)
    
    if not isinstance(persist_directory, str):
        vectordb = 'chromadb'
    
    for table, params in config.items():
        
        table = table.replace('$', '')
        
        params.append(model_name)
        params.append(vectordb)
        print(params)

        if 'sql_query' in table:
            setup_vector_db_sql_query(table, persist_directory, *params)
        elif table == 'company_name_chroma':
            setup_vector_db_company_name(table, persist_directory, *params, **db_conn)
        elif table == 'category_ratio_chroma':
            setup_vector_db_ratio(table, persist_directory, *params, **db_conn)
        elif params[0] == 'map_category_code_universal':
            setup_vector_db_universal(table, persist_directory, *params, **db_conn)
        else:
            setup_vector_db_fs(table, persist_directory, *params, **db_conn)
        logging.info(f'{table} vectordb setup completed')
            
    
def delete_embedding_db():
    vector_db_path = os.path.join(current_dir, '../data')
    
    if os.path.exists('../data'):
        subprocess.run(['rm', '-rf', vector_db_path])
        logging.info("Local vector db deleted successfully")
    else:
        logging.info("Local vector db does not exist")
    
            
def delete_everything(conn):
    delete_tables(conn, FIIN_DELETE_ORDER)
    
    logging.info("All tables deleted successfully")
    
    # delete_embedding_db()
    
    

def check_embedding_server(embedding_server):
    
    uri = f"{embedding_server}/embed"
    logging.info(f"Checking embedding server at {uri}")
    
    try:
        payload = {'inputs': 'test'}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(uri, json=payload, headers=headers)
        
        logging.info(f"Response embedding server: {response}")
        if response.status_code == 200:
            return True
    except Exception as e:
        print(e)
        return False
    
    return False

            
def setup_everything(config: dict):
    
    db_conn = {
        'db_name': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
        
    }
    print(db_conn)
    
    # Check available embedding server
    
    embedding_server = os.getenv('EMBEDDING_SERVER_URL')
    local_model = os.getenv('EMBEDDING_MODEL')
    
    # if not (check_embedding_server(embedding_server) or config.get('openai', False) or not os.getenv('LOCAL_EMBEDDING')):
    #     raise ValueError("No available embedding server")
    
    print(os.path.join(current_dir, '../data/vector_db_vertical_local'))
    
    client = PersistentClient(path = os.path.join(current_dir, '../data/vector_db_vertical_local'), settings = Settings())
    
    # delete everything
    if config.get('force', False):
        delete_everything(db_conn)
    elif config.get('reset_vector_db', False):
        delete_embedding_db()
      
    
    if not config.get('force', False):
        setup_rdb(not config.get('ignore_rdb', False), FIIN_RDB_SETUP_CONFIG, **db_conn)
        logging.info("RDB setup completed")
    else:
        setup_rdb(False, FIIN_RDB_SETUP_CONFIG, **db_conn)
    
    
    # Check if embedding server is running, if not use local model
    if config.get('local', False):
        
        if check_embedding_server(embedding_server):
            logging.info("Embedding server is running")
            setup_vector_db(VERTICAL_VECTORDB_SETUP_CONFIG, client, embedding_server, **db_conn)
            
        elif os.getenv('LOCAL_EMBEDDING'):
            
            try:
                import torch
            
                logging.warning("Embedding server is not running, using local model")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = HuggingFaceEmbeddings(model_name=local_model, model_kwargs = {'device': device})
                setup_vector_db(VERTICAL_VECTORDB_SETUP_CONFIG, client, model, **db_conn)
                
            except Exception as e:
                logging.error("Configured local model is not available")
                raise e
    
    if config.get('openai', False):
        client_openai = PersistentClient(path = os.path.join(current_dir, '../data/vector_db_vertical_openai'), settings = Settings())
        setup_vector_db(VERTICAL_VECTORDB_SETUP_CONFIG, client_openai, **db_conn)
        logging.info("OpenAI Embedding setup completed")
    
    
    # bge 
    logging.info("Vector DB setup completed")
            
if __name__ == '__main__':
    
    env_path = os.path.join(current_dir, '../.env')
    dotenv.load_dotenv(env_path)

    db_conn = {
        'db_name': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
        
    }

    query = """
    	select 
		industry,
		ratio_code,
		data_mean
	from industry_financial_ratio
	where 
		ratio_code = 'BDR'
		and year = 2016 
--		and quarter = 0
		and industry = 'Banking'
"""
    
    result = execute_query(query, conn=db_conn)
    print(result)
    


    