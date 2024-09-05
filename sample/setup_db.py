import psycopg2
import pandas as pd

# Step 1: Connect to PostgreSQL
def connect_to_db(db_name, user, password, host='localhost', port='5432'):
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn

# Step 2: Create table if it doesn't exist
def create_table_if_not_exists(conn, table_name, columns):
    with conn.cursor() as cur:
        # Replace this with the appropriate table creation logic based on your CSV structure
        column_definitions = ', '.join([f'{col} TEXT' for col in columns])  # Assume all columns are TEXT, modify if needed
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {column_definitions}
            );
        """)
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
        
        conn.commit()

# Step 4: Load CSV and call the functions
def load_csv_to_postgres(csv_path, db_name, user, password, table_name, port):
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Connect to the PostgreSQL database
    conn = connect_to_db(db_name, user, password, port=port)
    
    try:
        # Create the table if it doesn't exist
        create_table_if_not_exists(conn, table_name, df.columns)
        
        # Upsert the data into the table
        upsert_data(conn, table_name, df)
    finally:
        conn.close()

# Example usage
csv_path = 'data.csv'
db_name = 'test_db'
user = 'postgres'
password = '12345678'
table_name = 'financial_statement'
port = '5433'

load_csv_to_postgres(csv_path, db_name, user, password, table_name, port)