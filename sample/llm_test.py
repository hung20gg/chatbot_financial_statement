from llm.llm.chatgpt import ChatGPT
from llm.llm_utils import get_code_from_text_response, get_json_from_text_response


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
    