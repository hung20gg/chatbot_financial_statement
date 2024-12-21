from .base import BaseAgent
from . import text2sql_utils as utils
from .text2sql_utils import Table
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
from typing import Any, List
from copy import deepcopy

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
    
    llm_responses: List = [] # All the responses from the LLM model
    history: List = [] # The conversation history
    llm: Any = Field(default=None) # The LLM model
    sql_llm: Any = Field(default=None) # The SQL LLM model
    sql_dict: dict = {} # The SQL dictionary
    
    suggest_table: List = [] # The suggested table for the task
    company_info: Table = None # The company information
    
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
        self.llm_responses = []
        self.suggest_table = []
        self.history = []
        self.company_info = None
        self.sql_dict = {}
        
        
    def simplify_branch_reasoning(self, task):
        """
        Simplify the branch reasoning response
        """
        
        assert self.max_steps > 0, "Max steps must be greater than 0"
        
        brief_database = self.prompt_config.BREAKDOWN_NOTE_PROMPT
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in financial statement and database management. You are tasked to break down the given task to {self.max_steps-1}-{self.max_steps} simpler steps. If time not mentioned, assume Q3 2024."
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
        
        messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        messages = utils.reformat_messages(messages)
        
        self.llm_responses.extend(messages)
        return get_json_from_text_response(response, new_method=True)['steps']
     
     
       
    def get_stock_code_and_suitable_row(self, task):
        """
        Prompt and get stock code and suitable row
        Input:
            - task: str
            - format: str
        Output:
            format = 'table':
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
            
        messages = utils.reformat_messages(messages)
        self.llm_responses.extend(messages)
            
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
        
        company_df = Table(table=company_df, description="Company Info")
        tables = []
        for title, df in dict_dfs.items():
            
            if df is None:
                continue
            
            table = Table(table=df, description=title)
            tables.append(table)
        return company_df, tables

    
    @staticmethod 
    def __flatten_list(list_of_str, prefix = "error"):
        text = ""
        for i, item in enumerate(list_of_str):
            text += f"{prefix} {i+1}: {item}\n\n"
    
    
    def __debug_sql(self, history, error_messages: List[str]):
        
        error_message = self.__flatten_list(error_messages, prefix="Error")
        
        new_query = f"You have some error in the previous SQL query:\n\n <log>\n\n{error_message}\n\n</log>\n\n. Please fix the error and try again."
        history.append(
            {
                "role": "assistant",
                "content": new_query
            }
        )
        
        response = self.sql_llm(history)
        if self.config.verbose:
            print(response)
        error_messages, execution_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)
        return response, error_messages, execution_table
    
    def debug_sql_code(self, history: List[dict], error_messages: List[str] = []):
        
        """
        The debug_sql_code method is designed to debug SQL queries by iteratively refining them up 
        to a maximum of three times. It uses the SQL language model to identify and fix errors in the 
        SQL queries.
        
        Parameters:

            history (List[dict]): A list of the conversation history, including previous SQL queries and responses.
        
        Returns:

            history (list): Updated conversation history with debugging attempts.
            error_messages (list): A list of error messages encountered during the debugging process.
            execution_tables (list): A list of execution tables generated during the debugging process.
        
        """
        
        all_error_messages = []
        execution_tables = []
        debug_messages = []
        
        count_debug = 1
        
        while count_debug < 3: # Maximum 3 times to debug
            
            logging.info(f"Debug SQL code round {count_debug}")
            response, error_messages, execute_table = self.__debug_sql(history, error_messages[-1])
            all_error_messages.extend(error_messages)
            execution_tables.extend(execute_table)
            
            history.append({
                "role": "assistant",
                "content": response
            })
            
            debug_messages.append(history[-1])
            
            # If there is no error, break the loop
            if len(error_messages) == 0:
                break
            count_debug += 1
        
        return debug_messages, all_error_messages, execution_tables
    
    @staticmethod
    def sql_dict_to_markdown(sql_dict):
        text = ""
        for key, value in sql_dict.items():
            text += f"**{key}** \n\n```sql\n\n{value}```\n\n"
        return text
    
    
    def reasoning_text2SQL(self, task: str, company_info: Table = None, suggest_table: List[Table] = []):
        
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
        if company_info is None or company_info.table.empty:
            stock_code_table = ""
        else:
            stock_code_table = utils.table_to_markdown(company_info)
        
        system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query. If time not mentioned, assume collecting data in Q3 2024.
    """
        database_description = self.prompt_config.OPENAI_SEEK_DATABASE_PROMPT
        
        RAG_sql = self.db.find_sql_query_v2(text=task, top_k=self.config.sql_example_top_k)
        few_shot_dict = dict()
        
        # Reduce the number of SQL examples
        for key, value in RAG_sql.items():
            if key not in self.sql_dict:
                few_shot_dict[key] = value
                self.sql_dict[key] = value
        
        few_shot = self.sql_dict_to_markdown(few_shot_dict)
        
        init_prompt = self.prompt_config.REASONING_TEXT2SQL_PROMPT.format(database_description = database_description, 
                                                                     task = task, 
                                                                     stock_code_table = stock_code_table, 
                                                                     suggestions_table = utils.table_to_markdown(suggest_table), 
                                                                     few_shot = few_shot)
        
        
        new_prompt = self.prompt_config.CONTINUE_REASONING_TEXT2SQL_PROMPT.format(task = task, 
                                                                                    stock_code_table = stock_code_table,
                                                                                    suggestions_table = utils.table_to_markdown(suggest_table), 
                                                                                    few_shot = few_shot)
        
        if len(self.history) == 0:
            self.history = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": init_prompt
                }
            ]
            temp_message = self.history.copy()
        else:
            self.history.append({
                "role": "user",
                "content": new_prompt
            })
            temp_message = [self.history[-1]]
            
        response = self.sql_llm(self.history)
        if self.config.verbose:
            print(response)
        
        # Execute SQL Query with TIR reasoning    
        error_messages = []
        execution_tables = []
        
        error_message, execution_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)
        
        error_messages.extend(error_message)
        execution_tables.extend(execution_table)
        
        self.history.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        temp_message.append(self.history[-1])
        
        # Self-debug the SQL code
        if self.config.self_debug and len(error_message) > 0:
            debug_messages, debug_error_messages, debug_execution_tables = self.debug_sql_code(self.history, error_message)
            
            error_messages.extend(debug_error_messages)
            execution_tables.extend(debug_execution_tables)
            
            self.history.extend(debug_messages)
            temp_message.extend(debug_messages)
            
        self.llm_responses.extend(utils.reformat_messages(temp_message))   
        
        return self.history, error_messages, execution_tables
    
        
    def branch_reasoning_text2SQL(self, task: str, steps: list[str], company_info, suggest_table):
        
        """
        Branch reasoning with Text2SQL 
        Instead of solving the task directly, it will break down the task into steps and solve each step
        
        Input:
            - task: str. The task to be solved, provided as a natural language string.
            - steps: list[str]. The steps to break down the task.
            - company_info. Information about the company relevant to the task.
            - suggest_table. The suggested table for the task.
        Output:
            - history: list.
            - error_messages: list.
            - execution_tables: list
        
        Future work:
            - Simulate with Monte Carlo Tree Search
        """
        
        stock_code_table = utils.table_to_markdown(company_info)
        look_up_stock_code = f"\n\nHere are the detail of the companies: \n\n{stock_code_table}"

        database_description = self.prompt_config.OPENAI_SEEK_DATABASE_PROMPT
        init_prompt = self.prompt_config.BRANCH_REASONING_TEXT2SQL_PROMPT.format(database_description = database_description, 
                                                                             task = task, 
                                                                             steps_string = steps_to_strings(steps), 
                                                                             suggestions_table = utils.table_to_markdown(suggest_table))
    
        
        if len(self.history) == 0:
            task_index = 1
            
            self.history = [
                {
                    "role": "system",
                    "content": "You are an expert in financial statement and database management. You will be asked to convert a natural language query into a PostgreSQL query."
                },
                {
                    "role": "user",
                    "content": init_prompt + '\n\n' + look_up_stock_code
                }
            ]
        else:
            task_index = len(self.history)
            self.history.append({
                "role": "user",
                "content": init_prompt + '\n\n' + look_up_stock_code
            })
            
        error_messages = []
        execution_tables = []
        
        previous_result = ""
        
        for i, step in enumerate(steps):
            logging.info(f"Step {i+1}: {step}")
            if i == 0:
                self.history[-1]["content"] += f"<instruction>\n\nThink step-by-step and do the {step}\n\n</instruction>\n\nHere are the samples SQL you might need\n\n{self.db.find_sql_query(step, top_k=self.config.sql_example_top_k)}\n\n"
            else:
                self.history.append({
                    "role": "user",
                    "content": f"The previous result of is \n\n<result>\n\n{previous_result}\n\n<result>\n\n <instruction>\n\nThink step-by-step and do the {step}\n\n</instruction>\n\nHere are the samples SQL you might need\n\n{self.db.find_sql_query(step, top_k=self.config.sql_example_top_k)}\n\n"
                })
            
            response = self.sql_llm(self.history)
            if self.config.verbose:
                print(response)
            
            # Add TIR to the SQL query
            error_message, execute_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)
            
            error_messages.extend(error_message)
            execution_tables.extend(execute_table)
            
            self.history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )
            
            
            # Self-debug the SQL code
            if self.config.self_debug and len(error_message) > 0:
                debug_messages, debug_error_messages, debug_execution_tables = self.debug_sql_code(self.history)
                
                self.history.extend(debug_messages)
                error_messages.extend(debug_error_messages)
                execution_tables.extend(debug_execution_tables)
                
                previous_result = utils.table_to_markdown(debug_execution_tables)
            
            else:
                previous_result = utils.table_to_markdown(execute_table)
            
            # Prepare for the next step
            company_info = utils.get_company_detail_from_df(execution_tables, self.db) # dataframe
            
            stock_code_table = utils.table_to_markdown(company_info)
            look_up_stock_code = f"\nHere are the detail of the companies: \n\n{stock_code_table}"
            self.history[task_index]["content"] = init_prompt + '\n\n' + look_up_stock_code
                   
        messages = utils.reformat_messages(self.history.copy())
        self.llm_responses.extend(messages) 
        
        return self.history, error_messages, execution_tables
    
    
    def update_suggest_data(self, company_info: Table, suggest_table: list[Table]):
        """
        Update the suggest data. Avoid duplicate suggestions and reduce prompt token
        """
        
        if self.company_info is None:
            self.company_info = company_info
        else:
            self.company_info.table, company_info.table = utils.join_and_get_difference(self.company_info.table, company_info.table)
        
        if len(self.suggest_table) == 0:
            self.suggest_table = suggest_table
        
        else:
            available_tables = [table.description for table in self.suggest_table]
            
            for table in suggest_table:
                if table.description not in available_tables:
                    self.suggest_table.append(table)
                
                else:
                    index = available_tables.index(table.description)
                    self.suggest_table[index].table, table.table = utils.join_and_get_difference(self.suggest_table[index].table, table.table)
                    
        return company_info, suggest_table
    
    
    
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
            
        company_info, suggest_table = self.get_stock_code_and_suitable_row(str_task)
        
        company_info, suggest_table = self.update_suggest_data(company_info, suggest_table)
        
        if not self.config.branch_reasoning:
            
            # If steps are broken down
            if len(steps) != 0:
                task += "\n\nBreak down the task into steps:\n\n" + steps_to_strings(steps)         
        
        
            self.history, error_messages, execution_tables = self.reasoning_text2SQL(task, company_info, suggest_table)
        else:
            self.history, error_messages, execution_tables = self.branch_reasoning_text2SQL(task, steps, company_info, suggest_table)

        tables = [company_info]
        tables.extend(suggest_table)
        
        tables = utils.prune_unnecessary_data_from_sql(deepcopy(tables), self.history)
        
        tables = utils.prune_null_table(tables) # Remove null table
        
        tables.extend(execution_tables)
        
        
        
        end = time.time()
        logging.info(f"Time taken: {end-start}s")
        return self.history, error_messages, tables