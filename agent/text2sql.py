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
    _latest_task_index: int = 0 # The latest task index

    max_debug_round: int = 3 # The maximum number of debugging rounds
    
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
        self._latest_task_index = 0
        self.max_debug_round = 3

    
    def get_latest_task(self):
        if len(self.history) == 0:
            return ""
        
        return self.history[self._latest_task_index]['content']
        
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
        
        # Error handling JSON response
        if not isinstance(json_response, dict):
            json_response = dict()

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
    
    
    def __debug_sql(self, history, error_messages: List[str], prefix = "Debug"):
        
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
        error_messages, execution_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose, prefix=prefix)
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
        
        count_debug = 0
        
        while count_debug < self.max_debug_round: # Maximum 3 times to debug
            
            logging.info(f"Debug SQL code round {count_debug}")
            response, error_messages, execute_table = self.__debug_sql(history, error_messages[-1], prefix=f"Debug Round {count_debug + 1}")
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
        count_sql = 0
        for key, value in sql_dict.items():
            text += f"**{key}** \n\n```sql\n{value}```"
            
            # Add new line
            count_sql += 1
            if count_sql < len(sql_dict):
                text += "\n\n"

        return text
    

    def get_reasoning_text2sql_template(self, task: str, company_info: Table = None, suggest_table: List[Table] = [], enhance: bool = False, adjust_table: str = 'shrink'):
        
        """
        Get the reasoning text2SQL template
        """
        
                # ============== Prepare the prompt =================
        if company_info is None or company_info.table.empty:
            stock_code_table = ""
        else:
            stock_code_table = utils.table_to_markdown(company_info, adjust = adjust_table)
        
        system_prompt = """
You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query. 
You will have the following database description:

### Database Description
{database_description}
"""
        
        # Add reagulation to the prompt
        if enhance:
            enhance_prompt = """
### Regulation:   
- Your should reasoning and provide the SQL query with explaination for the task in <task> and <instruction> tag.
- For <correction> tag, you only need to provide the SQL query if the previous SQL query is incorrect.
- For <reflection> tag, you dont need to provide the SQL query.
- Always return SQL query in the following format:
    ```sql
    {{SQL query}}
    ```
"""
            system_prompt += enhance_prompt

        database_description = self.prompt_config.OPENAI_SEEK_DATABASE_PROMPT.strip()
        system_prompt = system_prompt.format(database_description = database_description).strip()
        
        RAG_sql = self.db.find_sql_query_v2(text=task, top_k=self.config.sql_example_top_k)
        few_shot_dict = dict()
        
        # Reduce the number of SQL examples
        for key, value in RAG_sql.items():
            if key not in self.sql_dict:
                few_shot_dict[key] = value
                self.sql_dict[key] = value
        
        few_shot = self.sql_dict_to_markdown(few_shot_dict)
        
        init_prompt = self.prompt_config.REASONING_TEXT2SQL_PROMPT.format( 
                                                                     task = task, 
                                                                     stock_code_table = stock_code_table, 
                                                                     suggestions_table = utils.table_to_markdown(suggest_table, adjust = adjust_table), 
                                                                     few_shot = few_shot).strip()
        
        
        new_prompt = self.prompt_config.CONTINUE_REASONING_TEXT2SQL_PROMPT.format(task = task, 
                                                                                    stock_code_table = stock_code_table,
                                                                                    suggestions_table = utils.table_to_markdown(suggest_table, adjust = adjust_table), 
            
                                                                                few_shot = few_shot).strip()
        if enhance:
            enhance_prompt = (
                "\n\n"
                "Answer in the following format:\n"
                "### Reasoning:\n"
                "{Your reasoning}\n"
                "### SQL Query:\n"
                "{SQL query}"
            )

            init_prompt += enhance_prompt
            

        # =============================================


        # ================= Prune Tag =================
        if few_shot == "":

            init_prompt.replace("<example>\n", "")
            init_prompt.replace("</example>\n", "")

            new_prompt.replace("<example>\n", "")
            new_prompt.replace("</example>\n", "")

        init_prompt.replace('<data>\n\n\n</data>', "")
        init_prompt.replace('<data>\n\n\n\n\n</data>', "")
        new_prompt.replace('<data>\n\n\n</data>', "")
        new_prompt.replace('<data>\n\n\n\n\n</data>', "")

        # ============================================

        
        if len(self.history) == 0:
            temp_message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": init_prompt
                }
            ]
        else:
            temp_message = [
                {
                    "role": "user",
                    "content": new_prompt
                }
            ]            

        return temp_message
        
    
    def reasoning_text2SQL(self, task: str, company_info: Table = None, suggest_table: List[Table] = [], enhance: bool = False, inject_reasoning: str = None, adjust_table: str = 'shrink'):
        
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

        # ============== Prepare the prompt =================
        temp_message = self.get_reasoning_text2sql_template(task, company_info, suggest_table, enhance, adjust_table)
        self.history.extend(temp_message)

        self._latest_task_index = len(self.history) - 1

        # =========== Generate SQL query =============
        if inject_reasoning is not None: # Inject reasoning
            response = inject_reasoning
        else:
            response = self.sql_llm(self.history)

        if self.config.verbose:
            print(response)

        # ============================================
        

        # ===== Execute SQL Query with TIR reasoning =====
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

        # ===============================================
        

        # ========== Self-debug the SQL code ============
        if self.config.self_debug and len(error_message) > 0:
            debug_messages, debug_error_messages, debug_execution_tables = self.debug_sql_code(self.history, error_message)
            
            error_messages.extend(debug_error_messages)
            execution_tables.extend(debug_execution_tables)
            
            self.history.extend(debug_messages)
            temp_message.extend(debug_messages)
            
        # ===============================================

        self.llm_responses.extend(utils.reformat_messages(temp_message))   
        
        return self.history, error_messages, execution_tables


    def __flatten_sql_result(self, error_messages, execution_tables, adjust_table: str = 'keep'):
        sql_result = ""
        if len(execution_tables) > 0:
            sql_result += "### SQL Result from previous query:\n\n"
            sql_result += utils.table_to_markdown(execution_tables, adjust=adjust_table)
        if len(error_messages) > 0:
            sql_result += "### Error Messages from previous query:\n\n"
            for i, error in enumerate(error_messages):
                sql_result += f"{error}\n\n"
        return sql_result


    def self_correction(self, error_messages: list[str] = [], execution_tables: list[Table] = [], adjust_table: str = 'shrink'):
        
        """
        Self-correction with Text2SQL
        The self-correction method is designed to correct errors in SQL queries. It uses the SQL language model to identify and fix errors in the SQL queries.
        
        Parameters:

            task (str): The task to be solved, provided as a natural language string.
            error_messages (list): A list of error messages from SQL query.
            execution_tables (list): A list of execution tables generated during the process.
            cache (bool): A boolean value indicating whether to use the cache for the company information and suggested tables.
        
        Returns:

            history (list): A list of the conversation history.
            error_messages (list): A list of error messages from SQL query.
            execution_tables (list): A list of execution tables generated during the process.
            correct (bool): A boolean value indicating whether the previous SQL query has been corrected.
        
        """
        
        correction_prompt ="""
{sql_result}

<correction>

Based on the result, do you think the SQL query is correct and can fully answer the question? 

If the previous SQL not correct, return No under *Decision* heading, think step-by-step under *Reasoning* heading again and generate the correct SQL query under *SQL Query*.
Otherwise, you only need to return Yes under *Decision* heading. You must not provide the SQL query again.

Return in the following format (### SQL Query is optional):

### Decision: 
{{Your decision}}

### Reasoning:
{{Your reasoning}}

### SQL Query:
{{Corrected SQL query}}

</correction>
"""

        sql_result = self.__flatten_sql_result(error_messages, execution_tables, adjust_table)

        correction_prompt = correction_prompt.format(sql_result = sql_result)


        # ========= LLM Reasoning ========= #
        self.history.append({
            "role": "user",
            "content": correction_prompt
        })

        response = self.sql_llm(self.history)

        if self.config.verbose:
            print(response)

        self.history.append({
            "role": "assistant",
            "content": response
        })

        dict_response = utils.get_content_with_heading_tag(response, "###")
        
        router = True

        decision = dict_response.get("decision", "")

        if decision.lower().replace("{",'').replace("}",'').strip() in ["no", "n", "false", "f"]:
            router = False

            error_messages, execution_tables = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose)

        return self.history, error_messages, execution_tables, router



    def self_reflection(self, error_messages, execution_tables, cache: bool = True, adjust_table: str = 'shrink'):


        # ========= GET self-reflection ========= #
        reflection_prompt = """

{sql_result}

<reflection>

Based on the result, do you think the SQL query is correct and can fully answer the question? 
If not, return NO under *Decision* heading, and provide the reason and giving detailed tips to the correct SQL query under *Reflection* heading.
Else, you only need to return YES under *Decision* heading.

Only return new, detailed task. Do not return the SQL. Return in the following format:

### Decision: 
{{Your decision}}

### Reflection:
{{Your reflection}}

</reflection>
"""
        

        # ========= Router for self-reflection ========= #

        sql_result = self.__flatten_sql_result(error_messages, execution_tables)

        reflection_prompt = reflection_prompt.format(sql_result = sql_result)

        self.history.append({
            "role": "user",
            "content": reflection_prompt
        })

        reflection_response = self.sql_llm(self.history)

        if self.config.verbose:
            print(reflection_response)

        self.history.append({
            "role": "assistant",
            "content": reflection_response
        })

        

        reflection = False
        new_task = ""

        dict_response = utils.get_content_with_heading_tag(reflection_response, "###")

        decision = dict_response.get("decision", "")
        new_task = dict_response.get("reflection", "")

        if decision.lower().replace("{",'').replace("}",'').strip() in ["no", "n", "false", "f"]:
            print("Reflection is True")
            reflection = True
        else:
            reflection = False
            print("Reflection is False")

        if reflection:

            # ========== Get stock code and suitable row ========== #
            company_info, suggest_table = self.get_stock_code_and_suitable_row(new_task)
            
            tables = [company_info]
            tables.extend(suggest_table)

            if cache:
                company_info, suggest_table = self.update_suggest_data(deepcopy(company_info), deepcopy(suggest_table))
            # =================================================== #


            # ========= Reasoning with Text2SQL ========= #
            self.history, error_messages, execution_tables = self.reasoning_text2SQL(new_task, company_info, suggest_table, adjust_table = adjust_table)
            # =========================================== #

            return self.history, error_messages, execution_tables, reflection
        
        print("End of reflection")
        # No reflection
        return self.history, [], [], not reflection

    
        
    def branch_reasoning_text2SQL(self, task: str, steps: list[str], company_info, suggest_table, enhance: bool = False, adjust_table: str|int = 0):
        
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
        
        list_adj_table = ["shrink", "text", "keep"]
        if isinstance(adjust_table, int):
            choice = min(2, max(0, adjust_table))
            adjust_table = list_adj_table[choice]

        stock_code_table = utils.table_to_markdown(company_info, adjust=adjust_table)
        look_up_stock_code = f"\n\nHere are the detail of the companies: \n\n{stock_code_table}"

        database_description = self.prompt_config.OPENAI_SEEK_DATABASE_PROMPT
        init_prompt = self.prompt_config.BRANCH_REASONING_TEXT2SQL_PROMPT.format(database_description = database_description, 
                                                                             task = task, 
                                                                             steps_string = steps_to_strings(steps), 
                                                                             suggestions_table = utils.table_to_markdown(suggest_table, adjust=adjust_table))
    
        system_prompt = f"""
    You are an expert in financial statement and database management. You will be asked to convert a natural language query into a SQL query. If time not mentioned, assume collecting data in Q3 2024.
    You will have the following database description:

    ### Database Description

    {database_description}

    """
        
        if enhance:
            enhance_prompt = """
    ### Regulation        
    Your should reasoning and provide the SQL query with explaination for the task in <task>, <instruction> and/or <correction> tag.
    For <reflection> tag, you dont need to provide the SQL query.
    """
            system_prompt += enhance_prompt

        if len(self.history) == 0:
            task_index = 1
            
            self.history = [
                {
                    "role": "system",
                    "content": system_prompt
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

        self._latest_task_index = len(self.history) - 1
            
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
            error_message, execute_table = utils.TIR_reasoning(response, self.db, verbose=self.config.verbose, prefix=f"Step {i+1}")
            
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
                
                previous_result = utils.table_to_markdown(debug_execution_tables, adjust=adjust_table)
            
            else:
                previous_result = utils.table_to_markdown(execute_table, adjust=adjust_table)
            
            # Prepare for the next step
            company_info = utils.get_company_detail_from_df(execution_tables, self.db) # dataframe
            
            stock_code_table = utils.table_to_markdown(company_info, adjust=adjust_table)
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
    
    
    def refine_error_correction(self, error_messages, tables, refine_tool = "merge"):
        # Squeeze the error message into one single message and result
        if refine_tool == "merge":
            reasoning = f"<reasoning>\n\n{self.history[self._latest_task_index + 1]['content']}\n\n<reasoning>" # Most recent reasoning
            debugging = ""

            if self._latest_task_index + 1 < len(self.history) - 1: # If there is a debugging message
                for i in range(self._latest_task_index + 3, len(self.history)):
                    if self.history[i]['role'] == "assistant":
                        debugging += f"\n\n<debugging>\n\n{self.history[i]['content']}\n\n<debugging>"
            
            reasoning += debugging

            error_message = ""
            for i, prefix, error in enumerate(error_messages):
                error_message += f"{prefix}: {error}\n\n"

            table_message = utils.table_to_markdown(tables, adjust="shrink")

            return reasoning, error_message, table_message
        else:
            raise NotImplementedError(f"Refine tool {refine_tool} is not implemented")



    def get_solver_template_message(self, task: str, enhance: str = None, adjust_table: str|int = 0):

        list_adj_table = ["shrink", "text", "keep"]
        if isinstance(adjust_table, int):
            choice = min(2, max(0, adjust_table))
            adjust_table = list_adj_table[choice]

        bool_enhance = False
        if enhance is not None:
            bool_enhance = True
        
        company_info, suggest_table = self.get_stock_code_and_suitable_row(task)

        temp_message = self.get_reasoning_text2sql_template(task, company_info, suggest_table, bool_enhance, adjust_table)
        return temp_message

    
    def solve(self, task: str, cache: bool = True, inject_reasoning: str = None, enhance: str = None, adjust_table: str|int = 0):
        """
        Solve the task with Text2SQL
        The solve method is designed to solve a given task by converting it into SQL queries using the Text2SQL model. It handles both simple and complex tasks by breaking them down into steps if necessary.

        Parameters:

            task (str): The task to be solved, provided as a natural language string.
            cache (bool): A boolean value indicating whether to use the cache for the company information and suggested tables.
            inject_reasoning (str): If provided, the reasoning will be injected into the conversation history, instead of generating it from the model.

        Returns:

            history (list): A list of the conversation history.
            error_messages (list): A list of error messages from SQL query.
            execution_tables (list): A list of execution tables generated during the process.
            
        """
        list_adj_table = ["shrink", "text", "keep"]
        if isinstance(adjust_table, int):
            choice = min(2, max(0, adjust_table))
            adjust_table = list_adj_table[choice]

        bool_enhance = False
        if enhance is not None:
            self.max_debug_round = 1
            bool_enhance = True
        else:
            self.max_debug_round = 3
        
        start = time.time()

        # ===== Simplify the task with branch reasoning (Break down step) =====
        steps = []
        str_task = task
        if self.config.branch_reasoning or self.config.reasoning:
            steps = self.simplify_branch_reasoning(task)
            str_task = steps_to_strings(steps)


        # ===== Get stock code and suitable row =====    
        company_info, suggest_table = self.get_stock_code_and_suitable_row(str_task)
        
        tables = [company_info]
        tables.extend(suggest_table)

        if cache:
            company_info, suggest_table = self.update_suggest_data(deepcopy(company_info), deepcopy(suggest_table))
        

        # ===== Reasoning with Text2SQL =====
        if not self.config.branch_reasoning:
            
            # If steps are broken down
            if len(steps) != 0:
                task += "\n\nBreak down the task into steps:\n\n" + steps_to_strings(steps)         
        
        
            self.history, error_messages, execution_tables = self.reasoning_text2SQL(task, company_info, suggest_table, inject_reasoning = inject_reasoning, adjust_table = adjust_table, enhance = bool_enhance)
        else:
            self.history, error_messages, execution_tables = self.branch_reasoning_text2SQL(task, steps, company_info, suggest_table, adjust_table = adjust_table, enhance = bool_enhance)

        if bool_enhance:
            accept_result = False
            count_reflection = 0
            while not accept_result and count_reflection < 2: # Maximum 2 times to reflect
                count_reflection += 1
                logging.info(f"Reflection round {count_reflection}, using {enhance} method")

                if enhance == "correction":
                    self.history, error_messages, tables, accept_result = self.self_correction(error_messages, execution_tables, adjust_table=adjust_table)
                elif enhance == "reflection":
                    self.history, error_messages, tables, accept_result = self.self_reflection(error_messages, execution_tables, cache=cache, adjust_table=adjust_table)
                

        # ===== Post-processing =====

        tables = utils.prune_unnecessary_data_from_sql(tables, self.history)
        
        tables = utils.prune_null_table(tables) # Remove null table
        
        tables.extend(execution_tables)
        
        
        end = time.time()
        logging.info(f"Time taken: {end-start}s")
        return self.history, error_messages, tables