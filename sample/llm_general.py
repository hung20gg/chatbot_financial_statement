from llm.llm_utils import get_json_from_text_response, get_code_from_text_response
from setup_db import DBHUB
import utils


def get_stock_code_based_on_company_name(llm, task, db: DBHUB = None, top_k = 2, verbose=False):
    """
    Get the stock code based on the company name
    """
    
    messages = [
        {
            "role": "user",
            "content": f"""
Extract the company name based on the given question.
{task}
Only return exact the company name mentioned. Do not answer the question.
Return in JSON format. 

```json
{{
    "company_name": ["company1"]
}}
```
Return an empty list if no company name is found.
"""}]
    
    response = llm(messages)
    if verbose:
        print("Get stock code based on company name response: ")
        print(response)
        print("====================================")
    company_names = get_json_from_text_response(response, new_method=True)['company_name']
    if db is None:
        print("Not using DB")
        return company_names
    return utils.company_name_to_stock_code(db, company_names, top_k=top_k)


def find_suitable_column(llm, text, db: DBHUB = None, top_k=5, verbose=False):
    system_prompt = """
    You are an expert in analyzing financial reports. 
    """
    
    prompt = f"""
<thought>
{text}
</thought>

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
    
    if verbose:
        print("Find suitable column response: ")
        print(response)
        print("====================================")

    response = get_json_from_text_response(response)[0]    
    if db is None:
        return response
    
    bank_column = response.get("bank_column_name", [])
    non_bank_column = response.get("non_bank_column_name", [])
    
    return db.return_mapping_table(bank_column, non_bank_column, top_k)


def TIR_reasoning(response, db: DBHUB, verbose=False):
    codes = get_code_from_text_response(response)
    print(codes)
        
    TIR_response = ""
    execution_error = []
    execution_table = []
    
    for j, code in enumerate(codes):
        if code['language'] == 'sql':
            print(f"SQL Code {j+1}: \n{code['code']}")
            table = db.query(code['code'], return_type='dataframe')
            if isinstance(table, str):
                execution_error.append((j, table))
                continue
            execution_table.append(table)
            table_markdown = utils.df_to_markdown(table)
            TIR_response += f"SQL result for {j+1}: \n{table_markdown}\n\n"
    
    error_message = ""
    if len(execution_error) > 0:
        for i, error in execution_error:
            error_message += f"Error in SQL {i+1}: {error}\n\n"
            
    response += f"\n\n### The result of the given SQL:\n\n{TIR_response}"
    if len(error_message) > 0:
        for i, error in execution_error:
            response += f"\n\n### Error in SQL {i+1}:\n\n{error}"
    
    return response, error_message, execution_table