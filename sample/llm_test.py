from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
from llm_general import find_suitable_column, TIR_reasoning, get_stock_code_based_on_company_name
from setup_db import DBHUB
import utils
import re
import pandas as pd


def reasoning_text2SQL(llm, text, db: DBHUB, top_k = 5, verbose = False, running_type = 'sequential'):
    
    # Step 1: Find suitable column
    if running_type == 'parallel':
        bank_column = ""
        non_bank_column = ""
        company_info = ""
        pass 
    else:
        bank_column, non_bank_column = find_suitable_column(llm, text, db=db, top_k=top_k, verbose=verbose)
        company_info = get_stock_code_based_on_company_name(llm, text, db=db) 
        stock_code_table = utils.df_to_markdown(company_info)
               
    if verbose:
        print(f"Bank column: {bank_column}")
        print(f"Non-bank column: {non_bank_column}")
    
    # Step 2: Convert text to SQL
    system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
    
    database_description = utils.read_file_without_comments('prompt/seek_database.txt', start=['//'])
        
    few_shot = utils.read_file_without_comments('prompt/example1 no_col_name.txt')
        
    prompt = f"""You have the following database schema:
{database_description}

Here is a natural language query that you need to convert into a SQL query:
{text}

Company details
<data>
{stock_code_table}
</data>

Snapshot of the mapping table:
<data>
`map_category_code_bank`
{bank_column}

`map_category_code_non_bank`
{non_bank_column}
</data>

Here is an example of a query that you can refer to:

<example>
```sql
    {few_shot}
```
</example>

<instruction>
Think step-by-step and return SQL query that suitable with the database schema based on the natural language query above
</instruction>

Note: 
- Do not make any assumption about the column name. You can refer to the mapping table above to find the suitable column name.
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
    if verbose:
        print(response)
    
    # Add TIR to the SQL query
    response, error_message, execution_table = TIR_reasoning(response, db, verbose=verbose)
    
    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
    
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