from llm.llm.chatgpt import ChatGPT
from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
import re


class MappingTable:
    def __init__(self, vector_db_bank, vector_db_non_bank, db):
        self.vector_db_bank = vector_db_bank
        self.vector_db_non_bank = vector_db_non_bank
        self.db = db
        
    def search(self, texts, top_k, is_bank):
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
    
    def search_return_df(self, text, top_k, is_bank = False):
        collect_code = self.search(text, top_k, is_bank)
        collect_code = [f"'{code}'" for code in collect_code]
        query = f"SELECT category_code, en_caption FROM map_category_code_{'' if is_bank else 'non_'}bank WHERE category_code IN ({', '.join(collect_code)})"
        return self.db.query(query,return_type='dataframe')


def text2sql(llm, text):
    system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
    
    with open('prompt/seek_database.txt', 'r') as f:
        database_description = f.read()
        
    with open('prompt/example1.txt', 'r') as f:
        few_shot = f.read()
        
    prompt = f"""You have the following database schema:
    {database_description}
    
    Here is a natural language query that you need to convert into a SQL query:
    {text}
    
    Here is an example of a query that you can refer to:
    {few_shot}
    
    Think step-by-step and return SQL query that suitable with the database schema based on the natural language query above
    """
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm(messages)
    return get_code_from_text_response(response)


def partial_text2sql_1(llm, text):
    system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
    
    with open('prompt/seek_database.txt', 'r') as f:
        database_description = f.read()

    prompt = f"""You have the following database schema:
    {database_description}
    
    Here is a natural language query that you need to convert into a SQL query:
    {text}
    
    However, you have no information about the detail of the database, you only only have to create partial SQL query for database exploration.
    
    Think step-by-step and return partial SQL query based on the natural language query above.
    """
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm(messages)
    return get_code_from_text_response(response)


def find_suitable_column(llm, text):
    system_prompt = """
    You are an expert in analyzing financial reports. 
    """
    
    prompt = f"""
    {text}
    
    Based on given question, analyze and suggest the suitable column in the financial statement that can be used to answer the question.
    Notice that there are two types of financial statements: one for banks and one for non-banks cooperate.
    
    Analyze and return the suggested column names in JSON format.
    You don't need to return both bank and non-bank column names if you think only one type of column is suitable.
    ```json
    {{
        "bank_column_name": [],
        "non_bank_column_name": []
    }}
    ```
    """
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm(messages)
    print(response)
    return get_json_from_text_response(response)[0]


def reasoning_text2SQL(llm, text, search_func, top_k):
    
    # Step 1: Find suitable column
    extracted_column = find_suitable_column(llm, text)
    
    bank_column = ""
    non_bank_column = ""
    
    if "bank_column_name" in extracted_column and len(extracted_column["bank_column_name"]) > 0:
        bank_column = df_to_markdown(search_func(extracted_column["bank_column_name"], top_k, is_bank=True))
    
    if "non_bank_column_name" in extracted_column and len(extracted_column["non_bank_column_name"]) > 0:
        non_bank_column = df_to_markdown(search_func(extracted_column["non_bank_column_name"], top_k, is_bank=False))
    
    
    # Step 2: Convert text to SQL
    system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
    
    with open('prompt/seek_database.txt', 'r') as f:
        database_description = f.read()
        
    with open('prompt/question_query.txt', 'r') as f:
        few_shot = f.read()
        
    prompt = f"""You have the following database schema:
{database_description}

Here is a natural language query that you need to convert into a SQL query:
{text}

Snapshot of the mapping table:
`map_category_code_bank`
{bank_column}

`map_category_code_non_bank`
{non_bank_column}

Here is an example of a query that you can refer to:
{few_shot}
    
Think step-by-step and return SQL query that suitable with the database schema based on the natural language query above
"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm(messages)
    return get_code_from_text_response(response)

def df_to_markdown(df):
    markdown = df.to_markdown(index=False)
    return markdown



def CoT_reasoning(llm, text, query):
    system_prompt = """
    You are an expert in financial statement and database management. You are given a question a partial result of some financial statements and you need to 
    given step-by-step reasoning for the given question.
    """
    
    prompt = f"""
    You are asked to provide step-by-step reasoning for the following question:
    {text}
    
    The partial result of the financial statement is as follows:
    {str(query)}
    
    Reasoning for the given question based on the partial result of the financial statement step-by-step
    """
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm(messages)
    return response

 
def replace_category_query(sql_query, search_func, top_k): # Passing a function that returns top k categories
    # Pattern to match `category = xx` or `category = 'xx'`
    pattern_equal = re.compile(r"category\s*=\s*'?(\w+)'?")
    # Pattern to match `category IN (xx, yy)` or `category IN ('xx', 'yy')`
    pattern_in = re.compile(r"category\s+IN\s*\(([^)]+)\)")
    
    # Check for `category = xx`
    if pattern_equal.search(sql_query):
        sql_query = pattern_equal.sub(lambda match: f"category IN ({', '.join(search_func(match.group(1), top_k))})", sql_query)
    
    # Check for `category IN (xx, yy)`
    elif pattern_in.search(sql_query):
        sql_query = pattern_in.sub(lambda match: f"category IN ({', '.join(set([item for x in match.group(1).split(',') for item in search_func(x.strip(), top_k)]))})", sql_query)

    return sql_query