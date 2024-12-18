from .base import BaseAgent

from .const import ChatConfig
from . import text2sql_utils as utils
from .text2sql import Text2SQL
import sys
sys.path.append('..')
from llm.llm_utils import flatten_conversation, get_json_from_text_response



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
    sql_index: int = 0
    
    def __init__(self, config: ChatConfig, text2sql: Text2SQL, **kwargs):
        super().__init__(config = config, text2sql = text2sql, **kwargs)
        
        self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        self.routing_llm = utils.get_llm_wrapper(model_name=config.routing_llm, **kwargs)
        self.setup()
        
    def setup(self):
        
        self.history = []
        self.display_history = []
        self.sql_index = 0
        
        system_instruction = """
You are a financial analyst and you have access to a database of financial statements of companies from 2019 to 2024.
Only confident to answer questions related to finance and accounting.

Note: the money unit is Million VND. Do not discuss about number format (e.g. 1e9).

"""
# Only answer questions related to finance and accounting.
# If the question is not related to finance and accounting, say You are only allowed to ask questions related to finance and accounting.
        self.history.append(
            {
                'role': 'system',
                'content': system_instruction
            }
        )
        
        
        
    def routing(self, user_input):
        routing_log = deepcopy(self.display_history)
        if len(routing_log) < 1:
            routing_log = []
        
        routing_log.append(
            {
                'role': 'user',
                'content': f"""
    You are having a database stored financial statement of companies, and you are tasked to trigger it.
    You can only trigger the function only if the user question is related to financial reports and cannot be answered based on provided data.       
    
    Here is the user input:
    
    <input>
    
    {user_input}
    
    </input>
    
    Only accept the trigger if the user question can be answered with financial reports and cannot be answered based on provided data.
    
    Return your decision in JSON format.
        
        ```json
        {{
            "trigger": false
        }}
        ```
    """
            }
        )
        
        response = self.routing_llm(routing_log)
        routing = get_json_from_text_response(response, new_method=True)['trigger']
        return routing
    
    
    def summarize_and_get_task(self, messages):
        short_messages = messages[self.sql_index:][-5:] # Get the last 5 messages
        
        task = short_messages[-1]['content']
        short_messages.pop()
        
        prompt = [
            {
                'role': 'system',
                'content': "You have a financial report database. You are given the conversation history and you are tasked to get the most current SQL-related task from the conversation. Analyze and return the most current task in human language. Do not return SQL"
            },
            {
                'role': 'user',
                'content': f"Here is the conversation history\n\n{flatten_conversation(short_messages)}. Here is the current request\n\n{task}"
                            
            }
        ]
        
        response = self.routing_llm(prompt)
        return response
        
    
        
    def __reasoning(self, user_input):
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        try:
            routing = self.routing(user_input)
        except Exception as e:
            logging.error(f"Routing error: {e}")
            routing = False
        
        table_strings = ""
        if routing:
            logging.info("Routing triggered")
            
            task = user_input
            if len(self.history) > self.sql_index + 2: # If there are more than a conversation   
                logging.info("Summarizing and getting task")
                task = self.summarize_and_get_task(self.history)
            
            self.sql_history, error_messages, execution_tables =  self.text2sql.solve(task, history=self.sql_history)
            with open('sql_history.json', 'w') as file:
                json.dump(self.text2sql.llm_responses, file)
            
            
            for i, table in enumerate(execution_tables):
                table_strings += f"Table {i+1}: {utils.table_to_markdown(table)}\n\n"
            
            self.history.append(
                {
                    'role': 'user',
                    'content': f'You are provided with the following data:\n\n<table>\n\n{table_strings}\n\n<table>\n\nAnalyze and answer the following question:\n\n<input>\n\n{user_input}\n\n<input>\n\nYou should provide the answer based on the provided data, ignore if some data is missing.'
                }
            )
            self.sql_index = len(self.history)
        
        else:
            logging.info("Routing not triggered")
            self.history.append(
                {
                    'role': 'user',
                    'content': user_input
                }
            )
        return table_strings
        
        # return response
        
    def stream(self, user_input):
        start = time.time()
        
        table_strings = self.__reasoning(user_input)
        yield table_strings
        yield '\n\nAnalyzing\n\n'
        
        
        end = time.time()
        logging.info(f"Reasoning time with streaming: {end - start}s")
        
        # return self.llm.stream(self.history)
        response = self.llm.stream(self.history)
        text_response = []
        for chunk in response:
            # self.get_generated_response(response)
            yield chunk # return the response
            if isinstance(chunk, str):
                text_response.append(chunk)
            
        self.get_generated_response(''.join(text_response))
        
            
        
    def chat(self, user_input):
        start = time.time()
        
        self.__reasoning(user_input)
        response = self.llm(self.history)
        
        end = time.time()
        logging.info(f"Reasoning time without streaming: {end - start}s")
        
        self.get_generated_response(response)
        return response
    
    
    
    def get_generated_response(self, assistant_response):
        self.display_history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        self.history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        