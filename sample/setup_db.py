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
def create_table_if_not_exists(conn, table_name, df):
    columns = df.columns
    col_type = []
    
    for type_ in df.dtypes.values:
        if type_ == 'int64':
            col_type.append('INTEGER')
        elif type_ == 'float64':
            col_type.append('FLOAT')
        else:
            col_type.append('TEXT')

    with conn.cursor() as cur:
        # Replace this with the appropriate table creation logic based on your CSV structure
        
        column_definitions = ', '.join([f'{col} {type_}' for col,type_ in zip(columns, col_type)])  # Assume all columns are TEXT, modify if needed
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
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
def load_csv_to_postgres(csv_path, db_name, user, password, table_name, port):
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Connect to the PostgreSQL database
    conn = connect_to_db(db_name, user, password, port=port)
    
    try:
        # Create the table if it doesn't exist
        print('Creating table in database...')
        create_table_if_not_exists(conn, table_name, df)
        
        # Upsert the data into the table
        print('Upserting data into the table...')
        upsert_data(conn, table_name, df)
    finally:
        print('Closing connection to database...')
        conn.close()
        
        
def execute_query(query, conn = None):
    if conn is None:
        conn = connect_to_db(db_name, user, password, host, port)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
    finally:
        conn.close()
    return result

## SET UP EMBEDDING DATABASE

def create_chroma_db(collection_name, persist_directory):
    embedding_function = OpenAIEmbeddings(api_key = os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
    
    return Chroma(collection_name = collection_name, 
                  embedding_function = embedding_function, 
                  persist_directory = persist_directory)
    
# WE only setup 1 cache RAG database for a table
def setup_chroma_db(db_name, user, password, host, port, collection_name, persist_directory):
    conn = connect_to_db(db_name, user, password, host, port)
    print("Connected to database")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT category FROM financial_statement fs2 ")
            categories = cur.fetchall()
            categories = [category[0] for category in categories]        
    finally:
        conn.close()
    print(categories)
    chroma_db = create_chroma_db(collection_name, persist_directory)
    
    for i in range(0, len(categories), 10):
        batch_categories = categories[i:i+10]
        chroma_db.add_texts(batch_categories)
    

# Example usage
if __name__ == '__main__':
    csv_path = 'data.csv'
    db_name = 'test_db'
    user = 'postgres'
    password = '12345678'
    table_name = 'financial_statement'
    port = '5433'
    host = 'localhost'

    collection_chromadb = 'test_chroma_db'
    persist_directory = 'data/test_chroma_db'

    # Load csv data to PostgreSQL
    #load_csv_to_postgres(csv_path, db_name, user, password, table_name, port)

    # Generate embeddings for the data
    setup_chroma_db(db_name, user, password, host, port, collection_chromadb, persist_directory)
        