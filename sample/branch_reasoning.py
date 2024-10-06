from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
from utils import read_file_without_comments
import numpy as np
import pandas as pd

def company_name_to_stock_code(db, names, method = 'similarity') -> pd.DataFrame:
    """
    Get the stock code based on the company name
    """
    if not isinstance(names, list):
        names = [names]
    
    if method == 'similarity': # Using similarity search
        stock_codes = []
        company_names = []
        for name in names:
            result = db.similarity_search(name, top_k=2)
            for item in result:
                stock_codes.append(item.metadata['stock_code'])
                company_names.append(item.page_content)
        result = pd.DataFrame({'stock_code': stock_codes, 'company_name': company_names})
        result = result.drop_duplicates(subset=['stock_code'])
        return result
    
    else: # Using rational DB
        dfs = []
        query = f"SELECT * FROM company WHERE company_name LIKE '%{name}%'"
        
        if method == 'bm25-ts':
            query = f"SELECT stock_code, company_name FROM company_info WHERE to_tsvector('english', company_name) @@ to_tsquery('{name}');"
        
        elif 'bm25' in method:
            pass # Using paradeDB
        
        else:
            raise ValueError("Method not supported")  
        
        for name in names:
            
            # Require translate company name in Vietnamese to English
            name = name # translate(name, 'vi', 'en')
            
            result = db.query(query, return_type='dataframe')
            
            dfs.append(result)
            
        if len(dfs) > 0:
            result = pd.concat(dfs)
        else:
            result = pd.DataFrame()
        return result
        


def simplify_branch_reasoning(llm, task, num_steps=3):
    """
    Breaks down the task into simpler steps
    """
    assert num_steps > 0, "num_steps should be greater than 0"
    messages = [
        {
            "role": "system",
            "content": f"You are tasked to break down the given task to {num_steps-1}-{num_steps+1} simpler steps. Please provide the steps."
        },
        {
            "role": "user",
            "content": f"""
You are a financial analyst at a company. You are tasked to break down the shareholders' question into simpler steps.   
<question>
Question: {task}
</question>
Here are some information you might need:        
{read_file_without_comments("prompt/breakdown_note.txt")}   
Thinking and return the steps in JSON format.
    ```json
    {
        "steps" : ["Step 1", "Step 2"]
    }
    ```         
"""
        }  
    ]
    
    response = llm(messages)
    return get_json_from_text_response(response, new_method=True)



def get_stock_code_based_on_company_name(llm, task):
    """
    Get the stock code based on the company name
    """
    
    messages = [
        {
            "role": "user",
            "content": f"""
Extract the company name based on the given question.
{task}

Return in JSON format.

```json
{{
    "company_name": ["company1", "company2"]
}}
```
Return an empty list if no company name is found.
"""}]
    
    response = llm(messages)
    return get_json_from_text_response(response, new_method=True)