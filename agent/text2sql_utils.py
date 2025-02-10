import pandas as pd
import numpy as np
import os

import sys 
sys.path.append('..')

from llm.llm.chatgpt import ChatGPT, OpenAIWrapper
from llm.llm.gemini import Gemini

from llm.llm_utils import get_code_from_text_response
from pydantic import BaseModel, ConfigDict
from typing import Union
import re
import copy

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Table(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    table: Union[pd.DataFrame, str, None]
    sql: str = ""
    description: str = ""
    
    def __str__(self):
        return f"Table(desc = {self.description}, num_rows = {len(self.table)}, num_columns = {len(self.table.columns)}"
    
    def __repr__(self):
        return f"Table(desc = {self.description}, num_rows = {len(self.table)}, num_columns = {len(self.table.columns)}"
    

def table_to_markdown(table: Table|pd.DataFrame|str|list, adjust:str = 'shrink', max_string = 5000) -> str:
    
    if table is None:
        return ""
    
    # If it's a string
    if isinstance(table, str):
        return table

    if not isinstance(table, list):
        table = [table]
        
    markdown = ""
    for i,t in enumerate(table):
        if isinstance(t, pd.DataFrame):
            if t.empty:
                continue
            markdown += df_to_markdown(t, adjust = adjust)[:max_string] + "\n\n"
        
        elif isinstance(t, Table):
            if t.table is None:
                continue
            markdown += f"**{t.description.strip()}**\n\n"
            markdown += df_to_markdown(t.table, adjust = adjust)[:max_string]
            if i < len(t.table) - 1:
                markdown += "\n\n"
            
        else:
            raise ValueError("Invalid table type")
    
    return markdown
    
    
def join_and_get_difference(df1, df2):
    
    main_cols = ''
    for col in df1.columns:
        if col in ['category_code', 'universal_code', 'stock_code', 'ratio_code']:
            main_cols = col
            break
    if main_cols == '':
        return df1, df2
    
    # If find main column
    
    diff = df2[~df2[main_cols].isin(df1[main_cols])]
    df1 = pd.concat([df1, diff])
    return df1, diff


def get_llm_wrapper(model_name, **kwargs):


    if '/' not in model_name: # Direct provider

        if 'gpt' in model_name:
            logging.info(f"Using ChatGPT with model {model_name}")
            return ChatGPT(model_name=model_name, **kwargs)
        
        elif 'gemini' in model_name:
            logging.info(f"Using Gemini with model {model_name}")
            return Gemini(model_name=model_name, random_key='exp' in model_name, **kwargs)
        
        elif 'deepseek' in model_name:
            logging.info(f"Found DeepSeek endpoint: {model_name}")
            host = os.getenv('DEEPSEEK_HOST')
            api_key = os.getenv('DEEPSEEK_API_KEY')
        else:
            raise ValueError("Model not supported")

    else: # Huggingface LLM
        host = os.getenv('LLM_HOST')
        api_key = os.getenv('LLM_API_KEY')

    logging.info(f"Using OpenAI Wrapper model: {model_name}  with host {host}")

    return OpenAIWrapper(host=host, api_key=api_key, model_name=model_name, **kwargs)
    


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
        return ''.join(new_lines)
    
def read_file(file_path):
    if not os.path.exists(file_path):
        Warning(f"File {file_path} not found")
        return ""
    
    with open(file_path, 'r') as f:
        return f.read()
    
    
def df_to_markdown(df, adjust:str = 'keep') -> str:
    if not isinstance(df, pd.DataFrame):
        return str(df)
    
    df = copy.deepcopy(df)
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    
    if adjust == 'text':
        columns = df.columns
        if num_cols > 2:
            logging.warning("Too many columns, Using shrink")
            return df_to_markdown(df, adjust='shrink')
        
        if num_cols == 1:
            text_df = f"List of items *{columns[0]}*\n"
            for i, row in df.iterrows():
                text_df += f"- {row[columns[0]]}"

                # Add new line if not the last row
                if i < num_rows - 1:
                    text_df += "\n"
            return text_df
        
        elif num_cols == 2:
            text_df = f"List of {columns[0]} with corresponding {columns[1]}\n"
            for i, row in df.iterrows():
                text_df += f"- {row[columns[0]]}: {row[columns[1]]}"

                # Add new line if not the last row
                if i < len(df) - 1:
                    text_df += "\n"
            return text_df
        
    if adjust == 'shrink':
        
        columns = df.columns
        text_df = "| "
        for col in columns:
            text_df += f"{col} | "
        text_df = text_df[:-1] + "\n|"
        for col in columns:
            text_df += " --- |"
        text_df += "\n"

        for i, row in df.iterrows():
            text_df += "| "
            for r in row:
                text_df += f"{r} | "

            # Add new line if not the last row
            if i < num_rows - 1:
                text_df += "\n"
        return text_df
    
    else:
        logging.warning("Adjust not supported")
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
    
    
def get_table_name_from_sql(sql_text):
    pattern = r"-- ###\s*(.+)"
    matches = re.findall(pattern, sql_text)
    if len(matches) > 0:
        return matches[0]
    return ""
    



def get_content_with_heading_tag(content: str, tag: str = "###") -> dict:
    
    pattern = tag
    pattern = tag + r"\s*(.*?)\s*:(.*?)\n(?=" + tag + r"|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    result = dict()
    if not matches:
        return result

    # Parse matches into a dictionary
    for key, value in matches:
        result[key.strip().lower().replace(" ", "_")] = value.strip() if value.strip() else None

    return result

    
def TIR_reasoning(response, db, verbose=False, prefix=""):
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
            name = get_table_name_from_sql(code)
              
            table = db.query(code, return_type='dataframe')
            
            # If it see an error in the SQL code
            if isinstance(table, str):
                execution_error.append(f"{prefix} SQL {i+1} Error: " + table)
                
            else:
                table_obj = Table(table=table, sql=code, description=f"{prefix} SQL {i+1} Result: {name}".strip())
                execution_table.append(table_obj)
    
    
    return execution_error, execution_table

    
def get_company_detail_from_df(dfs, db, method = 'similarity') -> pd.DataFrame:
    stock_code = set()
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    if isinstance(dfs[0], Table):
        dfs = [df.table for df in dfs]
    
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
    
 
def check_openai_response(messages):
    if len(messages) == 0:
        return False
    
    for message in messages:
        if message.get('role', '') not in ['assistant', 'system', 'user']:
            return False
    
    return True
    
    
def reformat_messages(messages):
    
    if not check_openai_response(messages):
        raise ValueError("Invalid messages")
    
    flag_system = False
    system_message = ""
    if messages[0].get('role','') == 'system':
        flag_system = True
        system_message = messages[0]['content']
        messages = messages[1:]
        
    new_messages = []
    for i, message in enumerate(messages):
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if i == 0 and flag_system:
            content =  f"<<SYS>>\n\n{system_message}\n\n<<SYS>>\n\n" + content
        
        new_messages.append({
            'role': role,
            'content': content
        })
    if new_messages[-1].get('role', '') == 'user':
        new_messages.append(
            {
                'role': 'assistant',
                'content': "Something went wrong, please try again"
            }
        )
        
    return new_messages
            
            
def _prune_entity(table: pd.DataFrame, entities: list[str]):
    entities = np.array(entities)
    if not isinstance(entities, list):
        entities = [entities]

    cols = table.columns
    table['mask'] = 0
    
    for col in cols:
        if col in ['is_bank','is_securities'] and col in table.columns:
            table.drop(col, axis=1, inplace=True)
            continue
            
        mask_contain_entities = np.isin(table[col].values, entities)
        table.loc[mask_contain_entities, 'mask'] += 1
        
    table = table[table['mask'] > 0]
    table = table.drop('mask', axis=1)
    
    return table
            
            
def prune_unnecessary_data_from_sql(tables: list[Table], messages: list[dict]): 
    is_list = True
    if not isinstance(tables, list):
        is_list = False
        tables = [tables]
        
    assistant_messages = []
    for message in messages:
        if message.get('role', '') == 'assistant':
            assistant_messages.append(message.get('content', ''))
            
    sql_codes = []
    for message in assistant_messages:
        codes = get_code_from_text_response(message)
        for code in codes:
            if code['language'] == 'sql':
                sql_codes.append(code['code'])
    
    mentioned_entities = set()
    
    for code in sql_codes:
        matches = re.findall(r"'(.*?)'", code)
        mentioned_entities.update(matches)
    
    for table in tables:

        # Allow successful query only
        if isinstance(table.table, pd.DataFrame):
            table.table = _prune_entity(table.table.copy(), list(mentioned_entities))

    if not is_list:
        return tables[0]
    return tables
    

def reconstruct_tables_from_sql(db, sql_message, description = ""):

    errors, tables = TIR_reasoning(sql_message, db)

    if len(errors) > 0:

        return []
    
    mapping_table = []

    # Get the company_info
    stock_code = []
    for table in tables:
        if 'stock_code' in table.table.columns:
            try:
                stock_code.extend(table.table['stock_code'].tolist())
            except:
                pass

    if len(stock_code) > 0:
        df_company_info = db.return_company_from_stock_codes(stock_code)
        mapping_table.append(Table(table=df_company_info, description="Company Information"))

    # Get the ratio_code

    ratio_code = []
    for table in tables:
        if 'ratio_code' in table.table.columns:
            try:
                ratio_code.extend(table.table['ratio_code'].tolist())
            except:
                pass
    
    set_ratio_code = set()
    for code in ratio_code:
        if isinstance(code, str):
            set_ratio_code.add(code)
    ratio_code = list(set_ratio_code)

    if len(ratio_code) > 0:
        df_ratio = db._get_mapping_ratio_from_ratio_codes(ratio_code)
        mapping_table.append(Table(table=df_ratio, description="Ratio Mapping"))

    # Get the category_code
    category_code = []
    for table in tables:
        if 'category_code' in table.table.columns:
            try:
                category_code.extend(table.table['category_code'].tolist())
            except:
                pass

    set_category_code = set()
    for code in category_code:
        if isinstance(code, str):
            set_category_code.add(code)
    category_code = list(set_category_code)

    if len(category_code) > 0:
        df_category = db._get_mapping_category_from_category_codes(category_code)
        mapping_table.append(Table(table=df_category, description="Category Mapping"))

    mapping_table.extend(tables)

    return mapping_table
    



def check_null_table(tables: Table|pd.DataFrame):
    
    if isinstance(tables, pd.DataFrame):
        if tables is None or tables.empty:
            return True
        return False
    
    if isinstance(tables, Table):
        if tables.table is None or tables.table.empty:
            return True
        return False
    
    
def prune_null_table(tables: list[Table|pd.DataFrame]):
    new_tables = []
    for table in tables:
        
        if check_null_table(table):
            continue
        new_tables.append(table)
        
    return new_tables



if __name__ == '__main__':
    

    from dotenv import load_dotenv
    load_dotenv()

    print(os.getenv('LLM_HOST'))

    llm = get_llm_wrapper('deepseek-chat')
    
    message = [
        {
            'role': 'user',
            'content': "What is the revenue of Apple in Q2 2023"
        }
    ]

    response = llm(message)
    print(response)