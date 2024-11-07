from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
from llm_general import find_suitable_row_v2, TIR_reasoning, get_stock_code_based_on_company_name, debug_SQL, get_stock_code_and_suitable_row
from setup_db import DBHUB
import utils
import numpy as np
import pandas as pd

import sys
import os
sys.path.append('..')
current_dir = os.path.dirname(__file__)

    # Get the detail of the company (Not done yet)

def simplify_branch_reasoning(llm, task, num_steps=2, verbose=False):
    """
    Breaks down the task into simpler steps
    """
    assert num_steps > 0, "num_steps should be greater than 0"
    messages = [
        {
            "role": "system",
            "content": f"You are an expert in financial statement and database management. You are tasked to break down the given task to {num_steps-1}-{num_steps} simpler steps. Please provide the steps."
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

<example>
### Task: ROA, ROE of all the company which are owned by VinGroup

Step 1: Find the stock code of the company that is owned by VinGroup in `df_sub_and_shareholders` table.
Step 2: Get ROA, ROE of the chosen stock codes in the `financial_ratio` table.

</example>

Note:
 - You should provide general steps to solve the question. 
 - You must not provide the SQL query. 
 - Each step should be a as independence as possible.
 - The number of steps should be lowest if possible. You will be heavily penalized if create meaningless steps
 - You must not provide the steps that are too obvious or easy for an SQL query (retrieve and data,..).
 
Based on the question and databse, thinking and return the steps in JSON format.
    ```json
    {{
        "steps" : ["Step 1"]
    }}
    ```         
"""
        }  
    ]
    
    response = llm(messages)
    if verbose:
        print("Branch reasoning response: ")
        print(response)
        print("====================================")
    return get_json_from_text_response(response, new_method=True)['steps']



def llm_branch_reasoning(llm, task, db: DBHUB, self_debug = False, verbose=False, sql_llm = None, get_all_table=False):

    """
    Branch reasoning for financial statement
    """
    if sql_llm is None:
        sql_llm = llm
    
    steps = simplify_branch_reasoning(llm, task, verbose=verbose)
    
    steps_string = ""
    for i, step in enumerate(steps):
        steps_string += f"Step {i+1}: \n {step}\n\n"
    
    # Check step 1: Extract company name

    print("Step 0: Extract company name")
    company_info_df, suggestions_table = get_stock_code_and_suitable_row(llm, steps_string, db=db, verbose=verbose, get_all_table=get_all_table)
    stock_code_table = utils.df_to_markdown(company_info_df)
    look_up_stock_code = f"\nHere are the detail of the companies: \n\n{stock_code_table}"
    
    content = f"""You have the following database schema:

<description>
{utils.read_file_without_comments('prompt/openai_seek_database.txt', start=['//'])}
</description>

Here is a natural language query that you need to convert into a query:
<query>
{task}
</query>    

Note:
- You must get the financial ratio data in `financial_ratio` table.
- Your SQL query must only access the database schema provided.
- In each step, you should only do the task that is required. Do not do the task of next step.
- Make the SQL query as simple and readable as possible. Utilize existing data from previous steps to avoid unnecessary query.
- You are penalized if generating wrong or meaningless SQL query 
- If the data provided is enough to answer the question, you don't have to return SQL query.
        
Here are the steps to break down the task:
<steps>
{steps_string}
</steps>      

Snapshot of the mapping table:
<data>
{suggestions_table}
</data>
"""

    original_content = content
    
    history = [
        {
            "role": "system",
            "content" : "You are an expert in financial statement and database management. You will be asked to convert a natural language query into a PostgreSQL query."
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    cur_step = 0
    # Get company stock code
    # Need to make a copy to add new company table everytimes the code find a new company
    
    history[-1]["content"] += look_up_stock_code
    
    error_messages = []
    execution_tables = []
        
    # Other steps
    for i, step in enumerate(steps):
        cur_step += 1
        print(f"Step {cur_step}: {step}")
        
        history.append({
            "role": "user",
            "content": f"<instruction>\nThink step-by-step and do the {step}\n</instruction>\n\nHere are the samples SQL you might need\n\n{db.find_sql_query(step)}"
        })
        
        # print("RAG for step: ", cur_step, db.find_sql_query(step))
        
        response = sql_llm(history)
        if verbose:
            print(f"Step {cur_step} response: ")
            print(response)
            print("====================================")
        # Check TIR 
        response, error_message, execute_table = TIR_reasoning(response, db, verbose=verbose)
        
        error_messages.extend(error_message)
        execution_tables.extend(execute_table)
        
        history.append({
            "role": "assistant",
            "content": response
        })
        
        # Self-debug the SQL code
        count_debug = 0
        if len(error_message) > 0 and self_debug:
            while count_debug < 3:
                
                response, error_message, execute_table = debug_SQL(response, db, verbose=verbose)
                error_messages.extend(error_message)
                execution_tables.extend(execute_table)
            
                history.append({
                    "role": "assistant",
                    "content": response
                })
                if len(error_message) == 0:
                    break
                    
                count_debug += 1
          
        # Update the company info  
        new_company_info_df = utils.get_company_detail_from_df(execution_tables, db)
        if isinstance(new_company_info_df, pd.DataFrame):
            company_info_df = pd.concat([company_info_df, new_company_info_df])
            stock_code_table = utils.df_to_markdown(company_info_df)
            history[1]["content"] = original_content + f"\nHere are the detail of the companies: \n\n{stock_code_table}"
            
        else:
            # Need a logger here
            print("Error on execute SQL")
            
    return history, error_messages, execution_tables
        
