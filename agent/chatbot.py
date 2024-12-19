from .base import BaseAgent

from .const import ChatConfig
from . import text2sql_utils as utils
from .text2sql import Text2SQL
import sys
sys.path.append('..')
from llm.llm_utils import flatten_conversation, get_json_from_text_response, get_code_from_text_response
from ETL.mongodb import BaseSemantic


from pydantic import SkipValidation, Field
from typing import Any, Union, List
from copy import deepcopy
import logging
import time
import json
import uuid
import os

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
You are a financial analyst and you have access to a database of financial statements of companies from 2019 to Q3 2024.
Only confident to answer questions related to finance and accounting.

The provided data often in the form of tables inside <table> tag. However, the user cannot see the table in the prompt, so answer the question as natural as possible.

Note: the money unit is Million VND. Do not discuss about number format (e.g. 1e9 or percentage approximation). Always multiply ratio by 100 to get percentage and do not show this calculation step.

"""
# Only answer questions related to finance and accounting.
# If the question is not related to finance and accounting, say You are only allowed to ask questions related to finance and accounting.
        self.history.append(
            {
                'role': 'system',
                'content': system_instruction
            }
        )
        
        
    def create_new_chat(self, **kwargs):
        self.setup()
    
        
    def routing(self, user_input):
        routing_log = deepcopy(self.display_history)
        if len(routing_log) < 1:
            routing_log = []
        
        routing_log.append(
            {
                'role': 'user',
                'content': f"""
    You are now tasked to trigger the function to collect data from financial reports.
    You can only trigger the function only if the user question is related to financial reports and cannot be answered based on previous conversation.       
    
    Here is the user input:
    
    <input>
    
    {user_input}
    
    </input>
    
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
        short_messages = messages[-5:] # Get the last 5 messages
        
        task = short_messages[-1]['content']
        short_messages.pop()
        
        prompt = [
            {
                'role': 'system',
                'content': """
        You have a financial statement database. 
        You are given the conversation history between user and ai assistance and you are tasked to get the most current SQL-related task from the conversation. 
        
        Return the most current task in English.
         
        If the time is not mentioned, assume Q3 2024. 
        
        Do not return SQL
        """
            },
            {
                'role': 'user',
                'content': f"Here is the conversation history\n\n{flatten_conversation(short_messages)}.\n\n Here is the current request from user\n\n{task}"
                            
            }
        ]
        
        response = self.routing_llm(prompt)
        return response
        
        
    def _solve_text2sql(self, task):
        
        table_strings = ""
        
        self.sql_history, error_messages, execution_tables =  self.text2sql.solve(task, history=self.sql_history)
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/sql_history.json', 'w') as file:
            json.dump(self.text2sql.llm_responses, file)
        
        
        for i, table in enumerate(execution_tables):
            table_strings += f"Table {i+1}: {utils.table_to_markdown(table)}\n\n"
        
        self.history.append(
            {
                'role': 'user',
                'content': f"""
            You are provided with the following data:
            
            <table>
            
            {table_strings}
            
            <table>
            
            Analyze and answer the following question:
            
            <input>
            
            {task}
            
            <input>
            
            You should provide the answer based on the provided data. 
            The data often has unclear column names, but you can assume the data is correct and relevant to the task.
            If the provided data is not enough, try your best.
            
            Answer the question as natural as possible. Answer based on user's language.
            
            """
            }
        )
        self.sql_index = len(self.history)
        
    
    def solve_text2sql(self, user_input):
        self._solve_text2sql(user_input)
        
        
    def __reasoning(self, user_input):
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        try:
            task = user_input
            if self.config.get_task:
                logging.info("Summarizing and getting task")
                task = self.summarize_and_get_task(self.display_history.copy())
            routing = self.routing(task)
            
        except Exception as e:
            logging.error(f"Routing error: {e}")
            routing = False
        
        table_strings = ""
        if routing:
            logging.info("Routing triggered")
            
            self.solve_text2sql(task)
        
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
        
        
class ChatbotSematic(Chatbot):
    
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
    
    message_saver: BaseSemantic
    
    conversation_id: str = ""
    
    def __init__(self, message_saver: BaseSemantic, **kwargs):
        super().__init__(message_saver= message_saver, **kwargs)
        
        # self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        # self.routing_llm = utils.get_llm_wrapper(model_name=config.routing_llm, **kwargs)
        # self.setup()
        
    def solve_text2sql(self, task):
        self._solve_text2sql(task)
        response = self.sql_history[-1]['content']  
        codes = get_code_from_text_response(response)
        sqls = []
        for code in codes:
            if code['language'] == 'sql':
                sqls.append(code['code'])
                
        self.message_saver.add_sql(self.conversation_id, task, sqls)
        
        
    def create_new_chat(self, user_id: str = "test_user"):
        self.setup()
        self.conversation_id = self.message_saver.create_conversation(user_id)
        
        
    def get_generated_response(self, assistant_response):
        self.display_history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        self.history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        self.message_saver.add_message(self.conversation_id, self.display_history, self.sql_history)
        
        
    