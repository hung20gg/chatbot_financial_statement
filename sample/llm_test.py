from llm.llm.chatgpt import ChatGPT
from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
import re

def text2sql(llm, text):
    system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
    
    with open('prompt/seek_database.txt', 'r') as f:
        database_description = f.read()
        
    prompt = f"""You have the following database schema:
    {database_description}
    
    Here is a natural language query that you need to convert into a SQL query:
    {text}
    
    Returned SQL query that suitable with the database schema based on the natural language query above
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


def CoT_reasoning(llm, text, query):
    system_prompt = """
    You are an expert in financial statement and database management. You are given a question a partial result of some financial statements and you need to 
    given step-by-step reasoning for the given question.
    """
    
    with open('prompt/seek_database.txt', 'r') as f:
        database_description = f.read()
        
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