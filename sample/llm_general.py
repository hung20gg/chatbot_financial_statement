from llm.llm_utils import get_json_from_text_response, get_code_from_text_response
from setup_db import DBHUB
import utils


def get_stock_code_based_on_company_name(llm, task, db: DBHUB = None, top_k = 2, verbose=False, get_industry=False):
    """
    Get the stock code based on the company name
    """
    
    messages = [
        {
            "role": "user",
            "content": f"""
Extract the company name and/or the industry that positively mentioned based on the given question.
<question>
{task}
</question>
Only return exact the company name mentioned. Do not answer the question.
Return in JSON format. 

```json
{{
    "industry": [],
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
    industries = response.get("industry", [])

    if db is None:
        print("Not using DB")
        if get_industry:
            return company_names, industries
        return company_names
    
    if get_industry:
        return utils.company_name_to_stock_code(db, company_names, top_k=top_k), industries
    
    return utils.company_name_to_stock_code(db, company_names, top_k=top_k)


def find_suitable_row_v2(llm, text, stock_code = [], db: DBHUB = None, top_k=5, verbose=False, format = 'dataframe'):
    system_prompt = """
    You are an expert in analyzing financial reports. You are given 2 database, finacial statements and pre-calculated pre-calculated financial performance ratios.
    """
    
    prompt = f"""
    <thought>
    {text}
    </thought>

    <task>
    Based on given question, analyze and suggest the industry and suitable rows (categories) in the financial statement and/or financial performance ratios that can be used to answer the question.
    Analyze and return the suggested rows' name in JSON format.
    </task>

    <formatting_example>
    ```json
    {{
        "industry": [],
        "financial_statement_row": [],
        "financial_ratio_row": []
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

    response = get_json_from_text_response(response, new_method=True)    
    if db is None:
        return response
    
    industry = response.get("industry", [])
    financial_statement_row = response.get("financial_statement_row", [])
    financial_ratio_row = response.get("financial_ratio_row", [])
    
    dict_dfs = db.return_mapping_table_v2(financial_statement_row = financial_statement_row, financial_ratio_row = financial_ratio_row, industry = industry, stock_code = stock_code, top_k =top_k)
    
    if format == 'dataframe':
        return dict_dfs.values()
    
    elif format == 'markdown':
        text = ""
        for title, df in dict_dfs.items():
            text += f"\n\nTable `{title}`\n\n{utils.df_to_markdown(df)}"
            
        return text
    else:
        raise ValueError("Format not supported")


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



def debug_SQL(llm, history, db, verbose=False):
    """
    Debug the SQL code
    """
    
    new_query = "You have some error in the previous SQL query. Please fix the error and try again."
    history.append({
        "role": "user",
        "content": new_query
    })
    
    # Re-run the previous step
    response = llm(history)
    if verbose:
        print("Error response: ")
        print(response)
        print("====================================")
        
    return TIR_reasoning(response, db, verbose=verbose)