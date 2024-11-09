from .base import BaseAgent

from .const import Config, Text2SQLConfig
from . import text2sql_utils as utils
import sys
sys.path.append('..')

from ETL.hub import DBHUB
from .text2sql import Text2SQL

from copy import deepcopy
import logging
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Chatbot(BaseAgent):
    def __init__(self, config: Config, text2sql_config: Text2SQLConfig, db: DBHUB, max_steps: int = 2, **kwargs):
        super().__init__(config)
        self.db = db
        self.text2sql = Text2SQL(text2sql_config, db, max_steps, **kwargs)
        
        self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        self.routing_llm = utils.get_llm_wrapper(model_name=config.routing_llm, **kwargs)
        
        self.summary_every = config.summary_every
        self.display_history = []
        self.history = []
        self.tables = []
        
    def routing(self, user_input):
        routing_log = deepcopy(self.history)
        routing_log.append(
            {
                'role': 'user',
                'content': """
    You are having a database stored financial reports of companies from 2019 to 2024, and you are tasked to trigger it.
    You can only trigger the function only if the user question is related to financial reports and cannot be answered based on provided data.       
    Analyze and return your decision in JSON format.
        
        ```json
        {
            "trigger": false
        }
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
                table_strings += f"Table {i+1}\n+{utils.df_to_markdown(table)}\n\n"
            
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
        
        return self.llm(self.history)
    
    def get_generated_response(self, assistant_response):
        self.display_history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        self.history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        