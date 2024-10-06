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
def create_table_if_not_exists(conn, table_name, df, primary_key=None, foreign_key: dict = {},long_text=True):
    columns = df.columns
    col_type = []
    
    if primary_key is None:
        primary_key = set()
    else:
        primary_key = set(primary_key)
    
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            col_type.append('INTEGER')
        elif pd.api.types.is_float_dtype(df[col]):
            col_type.append('FLOAT')
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type.append('TIMESTAMP')
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
def upsert_data(conn, table_name, df):
    with conn.cursor() as cur:
        # Define a placeholder for the insert values
        placeholders = ', '.join(['%s'] * len(df.columns))
        # Convert DataFrame to list of tuples
        data_tuples = [tuple(x) for x in df.to_numpy()]
        
        # Perform the upsert operation
        for row in data_tuples:
            upsert_query = f"""
                INSERT INTO {table_name} VALUES ({placeholders})
            """
            cur.execute(upsert_query, row)
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
            cur.execute(f"SELECT  stock_code, company_name, en_company_name, short_name, en_short_name,  FROM {table}")
            companies = cur.fetchall()
            companies = [(company[0], company[1], company[2], company[3], company[4]) for company in companies]
    finally:
        conn.close()
    
    chroma_db = create_chroma_db(collection_name, persist_directory)
    
    for company in companies:
        print(company)
        chroma_db.add_texts([company[0]], metadatas=[{'lang': 'vi', 'stock_code': company[1]}])
        chroma_db.add_texts([company[0]], metadatas=[{'lang': 'en', 'stock_code': company[2]}])
        chroma_db.add_texts([company[0]], metadatas=[{'lang': 'vi', 'stock_code': company[3]}])
        chroma_db.add_texts([company[0]], metadatas=[{'lang': 'en', 'stock_code': company[4]}])

# Example usage
if __name__ == '__main__':
    # Database connection details
    db_name = 'test_db'
    user = 'postgres'
    password = '12345678'
    port = '5433'
    host = 'localhost'
    
    
#     csv_path = 'sample_data/map_category_code_non_bank.csv'
#     table_name = 'map_category_code_non_bank'
#     collection_chromadb = 'category_non_bank_chroma'
#     persist_directory = 'data/category_non_bank_chroma'

#     # Load csv data to PostgreSQL

#     load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
#     print("Loaded map_category_code_non_bank")
#     setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
#     print("Setup Chroma DB for map_category_code_non_bank")
#     # Generate embeddings  for the data
    
    
#     csv_path = 'sample_data/map_category_code_bank.csv'
#     table_name = 'map_category_code_bank'
#     collection_chromadb = 'category_bank_chroma'
#     persist_directory = 'data/category_bank_chroma'

#     # Load csv data to PostgreSQL
#     load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, primary_key=['category_code'])
#     print("Loaded map_category_code_bank")
#     setup_chroma_db_fs(db_name, user, password, host, port, collection_chromadb, persist_directory, table_name)
#     print("Setup Chroma DB for map_category_code_bank")
    
    
#     # Load Bank Financial Report
#     csv_path = 'sample_data/bank_financial_report.csv'
#     table_name = 'bank_financial_report'
#     load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_bank(category_code)'})
#     print("Loaded bank_financial_report")
    
#     # Load Non Bank Financial Report
#     csv_path = 'sample_data/non_bank_financial_report.csv'
#     table_name = 'non_bank_financial_report'
#     load_csv_to_postgres(csv_path, db_name, user, password, table_name, port, foreign_key = {'category_code': 'map_category_code_non_bank(category_code)'})
#     print("Loaded non_bank_financial_report")

    # Paths for the uploaded CSV files
    csv_path_company_info = 'sample_data/df_company_info.csv'
    csv_path_sub_and_shareholder = 'sample_data/df_sub_and_shareholder.csv'

    # Table names
    table_name_company_info = 'company_info'
    table_name_sub_and_shareholder = 'sub_and_shareholder'

    # Primary and foreign key definitions
    primary_key_company_info = ['stock_code']
    primary_key_sub_and_shareholder = None

    # Load 'company_info' data into PostgreSQL
    load_csv_to_postgres(
        csv_path=csv_path_company_info,
        db_name=db_name,
        user=user,
        password=password,
        table_name=table_name_company_info,
        port=port,
        primary_key=primary_key_company_info
    )
    print("Loaded company_info table")
    
    # Setup Chroma DB for company_info

    # Load 'sub_and_shareholder' data into PostgreSQL with foreign key relationship
    load_csv_to_postgres(
        csv_path=csv_path_sub_and_shareholder,
        db_name=db_name,
        user=user,
        password=password,
        table_name=table_name_sub_and_shareholder,
        port=port,
    )
    print("Loaded sub_and_shareholder table")

    # # Setup Chroma DB for company_info
    # setup_chroma_db(
    #     db_name=db_name,
    #     user=user,
    #     password=password,
    #     host=host,
    #     port=port,
    #     collection_name='company_info_chroma',
    #     persist_directory='data/company_info_chroma',
    #     table=table_name_company_info
    # )
    # print("Setup Chroma DB for company_info")

    # # Setup Chroma DB for sub_and_shareholder
    # setup_chroma_db(
    #     db_name=db_name,
    #     user=user,
    #     password=password,
    #     host=host,
    #     port=port,
    #     collection_name='sub_and_shareholder_chroma',
    #     persist_directory='data/sub_and_shareholder_chroma',
    #     table=table_name_sub_and_shareholder
    # )
    # print("Setup Chroma DB for sub_and_shareholder")
