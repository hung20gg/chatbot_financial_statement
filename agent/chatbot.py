from .base import BaseAgent

from .const import ChatConfig
from . import text2sql_utils as utils
from .text2sql import Text2SQL
# import sys
# sys.path.append('..')



from pydantic import SkipValidation, Field
from typing import Any, Union, List
from copy import deepcopy
import logging
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Chatbot(BaseAgent):
    
    text2sql: Text2SQL
    config: ChatConfig
    
    llm: Any = Field(default=None) # The LLM model
    routing_llm: Any = Field(default=None) # The SQL LLM model
    
    history: List[dict] = []
    display_history: List[dict] = []
    sql_history: List[dict] = []
    sql_request: List = []
    
    tables: List = [] 
    
    def __init__(self, config: ChatConfig, text2sql: Text2SQL, **kwargs):
        super().__init__(config = config, text2sql = text2sql, **kwargs)
        
        self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        self.routing_llm = utils.get_llm_wrapper(model_name=config.routing_llm, **kwargs)
        self.setup()
        
    def setup(self):
        
        self.history = []
        self.display_history = []
        
        system_instruction = """
You are a financial analyst and you have access to a database of financial statements of companies from 2019 to 2024.
Only answer questions related to finance and accounting.

If the question is not related to finance and accounting, say You are only allowed to ask questions related to finance and accounting.
"""
        self.history.append(
            {
                'role': 'system',
                'content': system_instruction
            }
        )
        
    def routing(self, user_input):
        routing_log = deepcopy(self.history)
        routing_log.append(
            {
                'role': 'user',
                'content': f"""
    You are having a database stored financial statement of companies from 2019 to 2024, and you are tasked to trigger it.
    You can only trigger the function only if the user question is related to financial reports and cannot be answered based on provided data.       
    
    Here is the user input:
    
    <input>
    
    {user_input}
    
    </input>
    
    Only accept the trigger if the user question can be answered with financial reports and cannot be answered based on provided data.
    
    Analyze and return your decision in JSON format.
        
        ```json
        {{
            "trigger": false
        }}
        ```
    """
            }
        )
        
        response = self.routing_llm(routing_log)
        routing = utils.get_json_from_text_response(response, new_method=True)['trigger']
        return routing
        
    def get_response(self, user_input):
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        try:
            routing = self.routing(user_input)
        except Exception as e:
            logging.error(f"Routing error: {e}")
            routing = False
        
        if routing:
            logging.info("Routing triggered")
            history, error_messages, execution_tables =  self.text2sql.solve(user_input)

            table_strings = ""
            for i, table in enumerate(execution_tables):
                table_strings += f"Table {i+1}\n+{utils.table_to_markdown(table)}\n\n"
            
            self.history.append(
                {
                    'role': 'user',
                    'content': f'You are provided with the following data:\n\n{table_strings}\n\nAnalyze and answer the following question:\n\n{user_input}'
                }
            )
        
        else:
            logging.info("Routing not triggered")
            self.history.append(
                {
                    'role': 'user',
                    'content': user_input
                }
            )
        
        return self.llm.stream(self.history)
    
    def get_generated_response(self, assistant_response):
        self.display_history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        self.history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        