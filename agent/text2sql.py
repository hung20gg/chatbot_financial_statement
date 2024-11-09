from .base import BaseAgent
from . import text2sql_utils as utils
import sys 
sys.path.append('..')
from ETL.hub import DBHUB
from llm.llm.abstract import LLM
from llm.llm_utils import get_json_from_text_response, get_code_from_text_response
from .const import Text2SQLConfig
from . import const

import logging
import time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Text2SQL(BaseAgent):
    def __init__(self, config: Text2SQLConfig, db: DBHUB, max_steps: int = 2, **kwargs):
        super().__init__(config)
        
        self.db = db
        
        # LLM
        self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        if hasattr(config, 'sql_llm'):
            self.sql_llm = utils.get_llm_wrapper(model_name=config.sql_llm, **kwargs)
        else:
            self.sql_llm = config.llm
            
        # Reasoning
        self.reasoning = config.reasoning
        self.branch_reasoning = config.branch_reasoning
        self.max_steps = max_steps
        
        self.self_debug = config.self_debug
        
        self.history = []
        self.llm_responses = []
        
    def reset(self):
        self.history = []
        self.llm_responses = []
        
        
    def simplify_branch_reasoning(self, task):
        """
        Simplify the branch reasoning response
        """
        
        assert self.branch_reasoning, "Branch reasoning is not implemented"
        assert self.max_steps > 0, "Max steps must be greater than 0"
        
        brief_database = const.BREAKDOWN_NOTE_PROMPT
        
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in financial statement and database management. You are tasked to break down the given task to {num_steps-1}-{num_steps} simpler steps. Please provide the steps."
            },
            {
                "role": "user",
                "content": const.BREAKDOWN_TASK_PROMPT.format(task, brief_database)
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
            "content": const.GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT.format(task)
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
        financial_statement_row = json_response.get("financial_statement_row", [])
        financial_ratio_row = json_response.get("financial_ratio_row", [])
        
        
        # Get company data stock code
        company_df = utils.company_name_to_stock_code(self.db, company_names, top_k=self.config.company_top_k)
        stock_code = company_df['stock_code'].values
        
        # Get mapping table
        dict_dfs = self.db.return_mapping_table_v2(financial_statement_row = financial_statement_row, financial_ratio_row = financial_ratio_row, industry = industry, stock_code = stock_code, top_k =self.config.top_k, get_all_tables=self.config.get_all_table)    
        
        # Return data
        if format == 'dataframe':
            return company_df, dict_dfs.values()
        
        elif format == 'markdown':
            text = ""
            for title, df in dict_dfs.items():
                text += f"\n\nTable `{title}`\n\n{utils.df_to_markdown(df)}"
            return utils.df_to_markdown(company_df), text
        
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
    
    
    def reasoning_text2SQL(self, task: str, company_info: str, suggest_table: str, history: list = None):
        
        system_prompt = """
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query.
    """
        database_description = const.OPENAI_SEEK_DATABASE_PROMPT
        
        few_shot = self.db.find_sql_query(text=task, top_k=self.config.sql_top_k)
        
        prompt = const.REASONING_TEXT2SQL_PROMPT.format(database_description, task, company_info, suggest_table, few_shot)
        
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
            
        response = self.sql_llm(self.history)
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
        
    def branch_reasoning_text2SQL(self, task: str, company_info: str, suggest_table: str, history: list = None):
        raise NotImplementedError("Branch reasoning is not implemented")
    
    
    def solve(self, task: str):
        """
        Solve the task with Text2SQL
        """
        if self.branch_reasoning:
            steps = self.simplify_branch_reasoning(task)
            
        company_info, suggest_table = self.get_stock_code_and_suitable_row(task, format='markdown')
        
        if self.branch_reasoning_text2SQL:
            return self.branch_reasoning_text2SQL(task, company_info, suggest_table)
        
        return self.reasoning_text2SQL(task, company_info, suggest_table)
        
        
        
    