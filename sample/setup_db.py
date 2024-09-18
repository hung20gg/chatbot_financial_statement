import psycopg2
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import dotenv
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
def create_table_if_not_exists(conn, table_name, df, primary_key = None, foreign_key : dict = {}):
    columns = df.columns
    col_type = []
    
    if primary_key is None:
        primary_key = set()
    else:
        primary_key = set(primary_key)
    
    for type_ in df.dtypes.values:
        if type_ == 'int64':
            col_type.append('INTEGER')
        elif type_ == 'float64':
            col_type.append('FLOAT')
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
def upsert_data(conn, table_name, df):
    with conn.cursor() as cur:
        # Define a placeholder for the insert values
        placeholders = ', '.join(['%s'] * len(df.columns))
        # Convert DataFrame to list of tuples
        data_tuples = [tuple(x) for x in df.to_numpy()]
        
        # Perform the upsert operation
        for row in data_tuples:
            # Assuming the first column is the unique identifier for rewriting data
            upsert_query = f"""
                INSERT INTO {table_name} VALUES ({placeholders})

            """
            cur.execute(upsert_query, row)
            print(f'Upserted row: {row}')
        
        conn.commit()

# Step 4: Load CSV and call the functions
def load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key = None, foreign_key : dict = {}):
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
        
        
def execute_query(query, conn = None, return_type = 'tuple'):
    if conn is None:
        raise ValueError("Connection is not provided")
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            
            if return_type == 'dataframe':
                columns = [desc[0] for desc in cur.description]
                result = pd.DataFrame(result, columns=columns)
    except Exception as e:
        print(e)
        result = None
    return result

## SET UP EMBEDDING DATABASE

def create_chroma_db(collection_name, persist_directory):
    embedding_function = OpenAIEmbeddings(api_key = os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
    
    return Chroma(collection_name = collection_name, 
                  embedding_function = embedding_function, 
                  persist_directory = persist_directory)
    
# WE only setup 1 cache RAG database for a table
def setup_chroma_db(db_name, user, password, host, port, collection_name, persist_directory, table = 'map_category_non_bank'):
    conn = connect_to_db(db_name, user, password, host, port)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT vi_caption, en_caption, category_code category FROM {table} fs2 ")
            categories = cur.fetchall()
            categories = [(category[0], category[1], category[2]) for category in categories]        
    finally:
        conn.close()
    chroma_db = create_chroma_db(collection_name, persist_directory)
    
    for category in categories:
        print(category)
        chroma_db.add_texts([category[0]], metadatas=[{'lang': 'vi', 'code':category[2]}])
        chroma_db.add_texts([category[1]], metadatas=[{'lang': 'en', 'code':category[2]}])
    
class DB:
    def __init__(self, db_name, user, password, host, port):
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = connect_to_db(db_name, user, password, host, port)
    
    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.conn.close()
        
    def query(self, query, return_type = 'tuple'):
        return execute_query(query, self.conn, return_type)


# Example usage
if __name__ == '__main__':
   
    db_name = 'test_db'
    user = 'postgres'
    password = '12345678'
    port = '5433'
    host = 'localhost'

    csv_path = 'sample_data/map_category_code_non_bank.csv'
    table_name = 'map_category_code_non_bank'
    collection_chromadb = 'category_non_bank_chroma'
    persist_directory = 'data/category_non_bank_chroma'

    # Load csv data to PostgreSQL

    load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    print("Loaded map_category_code_non_bank")
    setup_chroma_db(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    print("Setup Chroma DB for map_category_code_non_bank")
    # Generate embeddings  for the data
    
    
    csv_path = 'sample_data/map_category_code_bank.csv'
    table_name = 'map_category_code_bank'
    collection_chromadb = 'category_bank_chroma'
    persist_directory = 'data/category_bank_chroma'

    # Load csv data to PostgreSQL
    load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
    print("Loaded map_category_code_bank")
    setup_chroma_db(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
    print("Setup Chroma DB for map_category_code_bank")
    
    
    # Load Bank Financial Report
    csv_path = 'sample_data/bank_financial_report.csv'
    table_name = 'bank_financial_report'
    load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_bank(category_code)'})
    print("Loaded bank_financial_report")
    
    # Load Non Bank Financial Report
    csv_path = 'sample_data/non_bank_financial_report.csv'
    table_name = 'non_bank_financial_report'
    load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_non_bank(category_code)'})
    print("Loaded non_bank_financial_report")
        