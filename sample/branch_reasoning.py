from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
from .setup_db import DBHUB
import utils
import numpy as np
import pandas as pd

def company_name_to_stock_code(db : DBHUB, names, method = 'similarity') -> pd.DataFrame:
    """
    Get the stock code based on the company name
    """
    if not isinstance(names, list):
        names = [names]
    
    if method == 'similarity': # Using similarity search
        return db.find_stock_code_similarity(names)
    
    else: # Using rational DB
        dfs = []
        query = "SELECT * FROM company WHERE company_name LIKE '%{name}%'"
        
        if method == 'bm25-ts':
            query = "SELECT stock_code, company_name FROM company_info WHERE to_tsvector('english', company_name) @@ to_tsquery('{name}');"
        
        elif 'bm25' in method:
            pass # Using paradeDB
        
        else:
            raise ValueError("Method not supported")  
        
        for name in names:
            
            # Require translate company name in Vietnamese to English
            name = name # translate(name, 'vi', 'en')
            query = query.format(name=name)
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
            "content": f"You are an expert in financial statement and database management. You are tasked to break down the given task to {num_steps-1}-{num_steps+1} simpler SQL query steps. Please provide the steps."
        },
        {
            "role": "user",
            "content": f"""
You are a financial analyst at a company. You are tasked to break down the shareholders' question into simpler steps.   
<question>
Question: {task}
</question>
Here are some information you might need:        
{utils.read_file_without_comments("prompt/breakdown_note.txt")}   
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



def llm_branch_reasoning(llm, task, db: DBHUB):
    """
    Branch reasoning for financial statement
    """
    
    steps = simplify_branch_reasoning(llm, task)['steps']
    steps_string = ""
    
    for i, step in enumerate(steps):
        steps_string += f"Step {i+1}: \n {step}\n\n"
    
    # Check step 1: Extract company name
    checkpoints_message = "Find the stock code of the listed company name"
    look_up_stock_code = ""
    
    history = [
        {
            "role": "system",
            "content" : "You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query."
        },
        {
            "role": "user",
            "content": f"""
You have the following database schema:
<description>
{utils.read_file_without_comments('prompt/seek_database.txt')}
</description>

Here is a natural language query that you need to convert into a SQL query:
<query>
{task}
</query>    

{find_suitable_column(llm, task, db=db)}  
        
Here are the steps to break down the task:
<steps>
{steps_string}
</steps>            
"""
        }
    ]
    
    cur_step = 1
    # Get company stock code
    if utils.edit_distance_score(steps[0].lower(), checkpoints_message.lower()) > 0.6:
        print("Step 1: Extract company name")
        company_names = get_stock_code_based_on_company_name(llm, task)
        stock_code_table = utils.df_to_markdown(company_name_to_stock_code(db, company_names))
        look_up_stock_code = f"For first step, here are the stock codes of the companies: \n{stock_code_table}"
        history[-1]["content"] += look_up_stock_code
        
        cur_step += 1
        steps = steps[1:]
        
    # Other steps
    for i, step in enumerate(steps):
        cur_step += 1
        print(f"Step {cur_step}: {step}")
        
        if i == 0:
            history[-1]["content"] += f"\nDo the Step {cur_step}: {step}"
        else:
            history.append({
                "role": "user",
                "content": f"Do the Step {cur_step}: {step}"
            })
        
        response = llm(history)
        
        # Check TIR 
        codes = get_code_from_text_response(response)
        
        TIR_response = ""
        
        for j, code in enumerate(codes):
            if code['language'] == 'sql':
                print(f"SQL Code {j+1}: \n{code['code']}")
                table = db.query(code['code'], return_type='dataframe')
                table_markdown = utils.df_to_markdown(table)
                TIR_response += f"SQL result for {j+1}: \n{table_markdown}\n\n"
                
        response += "\n\n" + TIR_response
        
        history.append({
            "role": "assistant",
            "content": response
        })
        
    return history
        
    
def find_suitable_column(llm, text, return_type='markdown', db: DBHUB = None, top_k=5):
    
    system_prompt = """
    You are an expert in analyzing financial reports. 
    """
    
    prompt = f"""
    {text}
    
    <task>
    Based on given question, analyze and suggest the suitable column in the financial statement that can be used to answer the question.
    Notice that there are two types of financial statements: one for banks and one for non-banks cooperate.
    
    Analyze and return the suggested column names in JSON format.
    You don't need to return both bank and non-bank column names if you think only one type of column is suitable.
    </task>
    
    <formatting_example>
    ```json
    {{
        "bank_column_name": [],
        "non_bank_column_name": []
    }}
    ```
    </formatting_example>
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
    
    extracted_column = get_json_from_text_response(response, new_method=True)
    
    if return_type == 'json':
        return extracted_column
    
    assert db is not None, "DBHUB object is required"
    bank_column = ""
    non_bank_column = ""
    
    if "bank_column_name" in extracted_column and len(extracted_column["bank_column_name"]) > 0:
        bank_column = utils.df_to_markdown(db.search_return_df(extracted_column["bank_column_name"], top_k, is_bank=True))
    
    if "non_bank_column_name" in extracted_column and len(extracted_column["non_bank_column_name"]) > 0:
        non_bank_column = utils.df_to_markdown(db.search_return_df(extracted_column["non_bank_column_name"], top_k, is_bank=False))
    
    snapshot = f"""
Snapshot of the mapping table:
<data>
`map_category_code_bank` 

{bank_column}

`map_category_code_non_bank` 

{non_bank_column}
</data>      
"""
    return snapshot
  
    
    
