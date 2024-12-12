import pandas as pd
import os

from llm.llm.chatgpt import ChatGPT, OpenAIWrapper
from llm.llm.gemini import Gemini

from llm.llm_utils import get_code_from_text_response

def get_llm_wrapper(model_name, **kwargs):
    if 'gpt' in model_name:
        return ChatGPT(model_name=model_name, **kwargs)
    
    elif 'gemini' in model_name:
        return Gemini(model_name=model_name, **kwargs)
    
    return OpenAIWrapper(model_name=model_name, **kwargs)
    


def read_file_without_comments(file_path, start=["#", "//"]):
    if not os.path.exists(file_path):
        Warning(f"File {file_path} not found")
        return ""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            if not any([line.startswith(s) for s in start]):
                new_lines.append(line)
        return '\n'.join(new_lines)
    
def read_file(file_path):
    if not os.path.exists(file_path):
        Warning(f"File {file_path} not found")
        return ""
    
    with open(file_path, 'r') as f:
        return f.read()
    
    
def df_to_markdown(df):
    if not isinstance(df, pd.DataFrame):
        return str(df)
    markdown = df.to_markdown(index=False)
    return markdown


def company_name_to_stock_code(db, names, method = 'similarity', top_k = 2) -> pd.DataFrame:
    """
    Get the stock code based on the company name
    """
    if not isinstance(names, list):
        names = [names]
    
    if method == 'similarity': # Using similarity search
        df = db.return_company_info(names, top_k)
        df.drop_duplicates(subset=['stock_code'], inplace=True)
        return df
    
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
            result = pd.DataFrame(columns=['stock_code', 'company_name'])
        return result
    
    
def is_sql_full_of_comments(sql_text):
    lines = sql_text.strip().splitlines()
    comment_lines = 0
    total_lines = len(lines)
    in_multiline_comment = False

    for line in lines:
        stripped_line = line.strip()
        
        # Check if it's a single-line comment or empty line
        if stripped_line.startswith('--') or not stripped_line:
            comment_lines += 1
            continue
        
        # Check for multi-line comments
        if stripped_line.startswith('/*'):
            in_multiline_comment = True
            comment_lines += 1
            # If it ends on the same line
            if stripped_line.endswith('*/'):
                in_multiline_comment = False
            continue
        
        if in_multiline_comment:
            comment_lines += 1
            if stripped_line.endswith('*/'):
                in_multiline_comment = False
            continue

    # Check if comment lines are the majority of lines
    return comment_lines >= total_lines  
    
    
    
def TIR_reasoning(response, db, verbose=False):
    codes = get_code_from_text_response(response)
        
    TIR_response = ""
    execution_error = []
    execution_table = []
    
    sql_code = []
    
    for code in codes:
        if code['language'] == 'sql':
            codes = code['code'].split(";")
            for content in codes:
                # clean the content
                if content.strip() != "":
                    sql_code.append(content)
            
    for i, code in enumerate(sql_code):    
        if verbose:    
            print(f"SQL Code {i+1}: \n{code}")
        
        if not is_sql_full_of_comments(code):    
            table = db.query(code, return_type='dataframe')
            
            # If it see an error in the SQL code
            if isinstance(table, str):
                execution_error.append((i, table))
                continue
            
            execution_table.append(table)
            table_markdown = df_to_markdown(table)
            TIR_response += f"SQL result for {i+1}: \n{table_markdown}\n\n"
    
    response += f"\n\n### The result of the given SQL:\n\n{TIR_response}"
    
    error_message = ""
    if len(execution_error) > 0:
        for i, error in execution_error:
            error_message += f"Error in SQL {i+1}: {error}\n\n"
            response += f"\n\n### Error in SQL {i+1}:\n\n{error}"
    
    return response, execution_error, execution_table

    
def get_company_detail_from_df(dfs, db, method = 'similarity') -> pd.DataFrame:
    stock_code = set()
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    for df in dfs:
        for col in df.columns:
            if col == 'stock_code':
                stock_code.update(df[col].tolist())
            if col == 'company_name':
                stock_code.update(company_name_to_stock_code(db, df[col].tolist(), method)['stock_code'].tolist())
            if col == 'invest_on':
                stock_code.update(company_name_to_stock_code(db, df[col].tolist(), method)['stock_code'].tolist())
            
    list_stock_code = list(stock_code)
    
    return company_name_to_stock_code(db, list_stock_code, method)
    