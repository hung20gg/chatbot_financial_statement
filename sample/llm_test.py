from llm.llm.chatgpt import ChatGPT


def text2sql(llm, text):
    system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
    
    with open('prompt/seek_database.txt', 'r') as f:
        database_description = f.read()
        
    
    
    