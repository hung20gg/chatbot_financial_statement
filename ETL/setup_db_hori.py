import pandas as pd
from sqlalchemy import create_engine

# Database connection details
db_user = 'postgres'  # Replace with your database username
db_password = '12345678'  # Replace with your database password
db_host = 'localhost'  # Replace with your database host, e.g., 'localhost'
db_port = '5433'  # Replace with your database port, e.g., '5432' for PostgreSQL
db_name = 'test_db'  # Replace with your database name

# Create a database engine
db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(db_url)

# List of CSV file paths to load
file_paths = [
    '../csv_horizontal/bank_financial_report_hori.csv',
    '../csv_horizontal/non_bank_financial_report_hori.csv',
    '../csv_horizontal/sec_financial_report_hori.csv',
    '../csv_horizontal/financial_ratios_hori.csv'
]

# Push data from CSV files to database
for file_path in file_paths:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Define table name based on the file name (you can customize this part)
    table_name = file_path.split('/')[-1].split('.')[0]  # Extract table name from file name

    # Push DataFrame to the SQL database
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data from {file_path} has been successfully pushed to the table '{table_name}' in the database.")

