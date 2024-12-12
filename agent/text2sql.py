from .base import BaseAgent
from . import text2sql_utils as utils
import sys 
sys.path.append('..')

from ETL.dbmanager import BaseDBHUB
from llm.llm.abstract import LLM
from llm.llm_utils import get_json_from_text_response, get_code_from_text_response
from .const import Text2SQLConfig, Config
from . import const
from .prompt.prompt_controller import PromptConfig, VERTICAL_PROMPT_BASE, VERTICAL_PROMPT_UNIVERSAL

import pandas as pd
import logging
import time
from pydantic import SkipValidation, Field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def steps_to_strings(steps):
    steps_string = "\nBreak down the task into steps:\n\n"
    for i, step in enumerate(steps):
        steps_string += f"Step {i+1}: \n {step}\n\n"
    return steps_string

class Text2SQL(BaseAgent):
    
    db: BaseDBHUB # The database connection.
    max_steps: int # The maximum number of steps to break down the task
    prompt_config: PromptConfig # The prompt configuration. This is for specify prompt for horizontal or vertical database design
    
    history: list = [] # The history of the conversation
    llm_responses: list = [] # All the responses from the LLM model
    
    llm: Any = Field(default=None) # The LLM model
    sql_llm: Any = Field(default=None) # The SQL LLM model
    
    def __init__(self, config: Config, prompt_config: PromptConfig, db, max_steps: int = 2, **kwargs):
        super().__init__(config=config, db = db, max_steps = max_steps, prompt_config = prompt_config)
        
        self.db = db
        self.max_steps = max_steps
        self.prompt_config = prompt_config
        
        # LLM
        self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        if hasattr(config, 'sql_llm'):
            self.sql_llm = utils.get_llm_wrapper(model_name=config.sql_llm, **kwargs)
        else:
            logging.warning("SQL LLM is not provided. Use the same LLM model for SQL")
            self.sql_llm = self.sql_llm

        
    def reset(self):
        self.history = []
        self.llm_responses = []
        
        
    def simplify_branch_reasoning(self, task):
        """
        Simplify the branch reasoning response
        """
        
        assert self.max_steps > 0, "Max steps must be greater than 0"
        
        brief_database = self.prompt_config.BREAKDOWN_NOTE_PROMPT
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in financial statement and database management. You are tasked to break down the given task to {self.max_steps-1}-{self.max_steps} simpler steps. Please provide the steps."
            },
            {
                "role": "user",
                "content": self.prompt_config.BRANCH_REASONING_PROMPT.format(task = task, brief_database = brief_database)
            }
        ]
    
        logging.info("Simplify branch reasoning response")
        response = self.llm(messages)
        if self.config.verbose:
            print("Branch reasoning response: ")
            print(response)
            print("====================================")
        
        self.llm_responses.append(response)
        return get_json_from_text_response(response, new_method=True)['steps']
     
     
       
    def get_stock_code_and_suitable_row(self, task, format = 'markdown'):
        """
        Prompt and get stock code and suitable row
        Input:
            - task: str
            - format: str
        Output:
            format = 'markdown':
                - company_info_df: str
                - suggestions_table: str
                
            format = 'dataframe':
                - company_info_df: pd.DataFrame
                - suggestions_table: [pd.DataFrame]
        """
        
        
        messages = [
        {
            "role": "user",
            "content": self.prompt_config.GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT.format(task = task)
        }]
        
        
        logging.info("Get stock code based on company name response")
        response = self.llm(messages)
        messages.append(
            {
                "role": "assistant",
                "content": response
            })
        if self.config.verbose:
            print("Get stock code based on company name response: ")
            print(response)
            print("====================================")
        self.llm_responses.append(messages)
            
        json_response = get_json_from_text_response(response, new_method=True)
        if self.db is None:
            return json_response
        
        # Get data from JSON response
        industry = json_response.get("industry", [])
        company_names = json_response.get("company_name", [])
        financial_statement_account = json_response.get("financial_statement_account", [])
        financial_ratio = json_response.get("financial_ratio", [])
        
        
        # Get company data stock code
        company_df = utils.company_name_to_stock_code(self.db, company_names, top_k=self.config.company_top_k)
        stock_code = company_df['stock_code'].values.tolist()
        
        # Get mapping table
        dict_dfs = self.db.return_mapping_table(financial_statement_row = financial_statement_account, 
                                                financial_ratio_row = financial_ratio, 
                                                industry = industry, 
                                                stock_code = stock_code, 
                                                top_k =self.config.account_top_k, 
                                                get_all_tables=self.config.get_all_acount)    
        
        # Return data
        if format == 'dataframe':
            return company_df, dict_dfs.values()
        
        elif format == 'markdown':
            text = ""
            for title, df in dict_dfs.items():
                text += f"\n\nTable `{title}`\n\n{utils.df_to_markdown(df)}"
            return company_df, text
        
        else:
            raise ValueError("Format not supported")
        
    def __debug_sql(self, history):
        
        new_query = "You have some error in the previous SQL query. Please fix the error and try again."
        history.append(
            {
                "role": "assistant",
                "content": new_query
            }
        )
        
        response = self.sql_llm(history)
        if self.config.verbose:
            print(response)
        return utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)
    
    def debug_sql_code(self, history):
        
        """
        The debug_sql_code method is designed to debug SQL queries by iteratively refining them up 
        to a maximum of three times. It uses the SQL language model to identify and fix errors in the 
        SQL queries.
        
        Parameters:

            history (list): A list of the conversation history, including previous SQL queries and responses.
        
        Returns:

            history (list): Updated conversation history with debugging attempts.
            error_messages (list): A list of error messages encountered during the debugging process.
            execution_tables (list): A list of execution tables generated during the debugging process.
        
        """
        
        error_messages = []
        execution_tables = []
        count_debug = 1
        
        while count_debug < 3: # Maximum 3 times to debug
            
            logging.info(f"Debug SQL code round {count_debug}")
            response, error_message, execute_table = self.__debug_sql(history)
            error_messages.extend(error_message)
            execution_tables.extend(execute_table)
            
            history.append({
                "role": "assistant",
                "content": response
            })
            
            # If there is no error, break the loop
            if len(error_message) == 0:
                break
            count_debug += 1
        
        return history, error_messages, execution_tables
    
    
    def reasoning_text2SQL(self, task: str, company_info: pd.DataFrame, suggest_table: str, history: list = None):
        
        """
        Reasoning with Text2SQL without branch reasoning.
        
        Input:
            - task: str. The task to be solved, provided as a natural language string.
            - company_info: pd.DataFrame. Information about the company relevant to the task.
            - suggest_table: str. The suggested table for the task.
            - history: list
        Output:
            - history: list.
            - error_messages: list.
            - execution_tables: list
            
        This function will convert the natural language query into SQL query and execute the SQL query
        """
        
        stock_code_table = utils.df_to_markdown(company_info)
        system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
        database_description = self.prompt_config.OPENAI_SEEK_DATABASE_PROMPT
        
        few_shot = self.db.find_sql_query(text=task, top_k=self.config.sql_example_top_k)
        
        prompt = self.prompt_config.REASONING_TEXT2SQL_PROMPT.format(database_description = database_description, task = task, stock_code_table = stock_code_table, suggestions_table = suggest_table, few_shot = few_shot)
        
        if history is None:
            history = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            history.append({
                "role": "user",
                "content": prompt
            })
            
        response = self.sql_llm(history)
        if self.config.verbose:
            print(response)
         
        
        # Execute SQL Query with TIR reasoning    
        error_messages = []
        execution_tables = []
        
        response, error_message, execution_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)
        error_messages.extend(error_message)
        execution_tables.extend(execution_table)
        
        history.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        
        # Self-debug the SQL code
        if self.config.self_debug:
            history, debug_error_messages, debug_execution_tables = self.debug_sql_code(history)
            error_messages.extend(debug_error_messages)
            execution_tables.extend(debug_execution_tables)
        
        self.llm_responses.append(history)    
        
        return history, error_messages, execution_tables
    
        
    def branch_reasoning_text2SQL(self, task: str, steps: list[str], company_info: pd.DataFrame, suggest_table: str, history: list = None):
        
        """
        Branch reasoning with Text2SQL 
        Instead of solving the task directly, it will break down the task into steps and solve each step
        
        Input:
            - task: str. The task to be solved, provided as a natural language string.
            - steps: list[str]. The steps to break down the task.
            - company_info: pd.DataFrame. Information about the company relevant to the task.
            - suggest_table: str. The suggested table for the task.
            - history: list
        Output:
            - history: list.
            - error_messages: list.
            - execution_tables: list
        
        Future work:
            - Simulate with Monte Carlo Tree Search
        """
        
        stock_code_table = utils.df_to_markdown(company_info)
        look_up_stock_code = f"\nHere are the detail of the companies: \n\n{stock_code_table}"

        database_description = self.prompt_config.OPENAI_SEEK_DATABASE_PROMPT
        content = self.prompt_config.BRANCH_REASONING_TEXT2SQL_PROMPT.format(database_description = database_description, task = task, steps_string = steps_to_strings(steps), suggestions_table = suggest_table)
    
        if history is None:
            task_index = 1
            
            history = [
                {
                    "role": "system",
                    "content": "You are an expert in financial statement and database management. You will be asked to convert a natural language query into a PostgreSQL query."
                },
                {
                    "role": "user",
                    "content": content + look_up_stock_code
                }
            ]
        else:
            task_index = len(history)
            history.append({
                "role": "user",
                "content": content + look_up_stock_code
            })
            
        error_messages = []
        execution_tables = []
        
        
        for i, step in enumerate(steps):
            logging.info(f"Step {i+1}: {step}")
            if i == 0:
                history[-1]["content"] += f"<instruction>\nThink step-by-step and do the {step}\n</instruction>\n\nHere are the samples SQL you might need\n\n{self.db.find_sql_query(step, top_k=self.config.sql_example_top_k)}"
            else:
                history.append({
                    "role": "user",
                    "content": f"<instruction>\nThink step-by-step and do the {step}\n</instruction>\n\nHere are the samples SQL you might need\n\n{self.db.find_sql_query(step, top_k=self.config.sql_example_top_k)}"
                })
            
            response = self.sql_llm(history)
            if self.config.verbose:
                print(response)
            
            # Add TIR to the SQL query
            response, error_message, execute_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)
            
            error_messages.extend(error_message)
            execution_tables.extend(execute_table)
            
            history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )
            
            
            # Self-debug the SQL code
            if self.config.self_debug:
                history, debug_error_messages, debug_execution_tables = self.debug_sql_code(history)
                error_messages.extend(debug_error_messages)
                execution_tables.extend(debug_execution_tables)
            
            
            # Prepare for the next step
            company_info = utils.get_company_detail_from_df(execution_tables, self.db)
            stock_code_table = utils.df_to_markdown(company_info)
            look_up_stock_code = f"\nHere are the detail of the companies: \n\n{stock_code_table}"
            history[task_index]["content"] = content + look_up_stock_code
                   
        self.llm_responses.append(history)
        return history, error_messages, execution_tables
    
    def solve(self, task: str):
        """
        Solve the task with Text2SQL
        The solve method is designed to solve a given task by converting it into SQL queries using the Text2SQL model. It handles both simple and complex tasks by breaking them down into steps if necessary.

        Parameters:

            task (str): The task to be solved, provided as a natural language string.

        Returns:

            history (list): A list of the conversation history.
            error_messages (list): A list of error messages from SQL query.
            execution_tables (list): A list of execution tables generated during the process.
            
        """
        
        start = time.time()
        steps = []
        str_task = task
        if self.config.branch_reasoning or self.config.reasoning:
            steps = self.simplify_branch_reasoning(task)
            str_task = steps_to_strings(steps)
            
        company_info, suggest_table = self.get_stock_code_and_suitable_row(str_task, format='markdown')
        
        if not self.config.branch_reasoning:
            
            # If steps are broken down
            if len(steps) == 0:
                task += "\nBreak down the task into steps:\n\n" + steps_to_strings(steps)         
        
        
            history, error_messages, execution_tables = self.reasoning_text2SQL(task, company_info, suggest_table)
        else:
            history, error_messages, execution_tables = self.branch_reasoning_text2SQL(task, steps, company_info, suggest_table)
        
        end = time.time()
        logging.info(f"Time taken: {end-start}s")
        return history, error_messages, execution_tables
    
    
    
        
        
        
    