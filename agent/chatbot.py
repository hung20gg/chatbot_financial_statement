from .base import BaseAgent

from .const import ChatConfig
from . import text2sql_utils as utils
from .text2sql import Text2SQL
import sys
sys.path.append('..')
from llm.llm_utils import flatten_conversation, get_json_from_text_response, get_code_from_text_response
from ETL.dbmanager import BaseSemantic

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

current_dir = os.path.dirname(os.path.realpath(__file__))

class Chatbot(BaseAgent):
    
    text2sql: Text2SQL
    config: ChatConfig
    
    llm: Any = Field(default=None) # The LLM model
    routing_llm: Any = Field(default=None) # The SQL LLM model
    
    history: List[dict] = []
    display_history: List[dict] = []
    sql_history: List[dict] = []
    
    tables: List = [] 
    is_routing: bool = False
    
    def __init__(self, config: ChatConfig, text2sql: Text2SQL, **kwargs):
        super().__init__(config = config, text2sql = text2sql, **kwargs)
        
        self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        self.routing_llm = utils.get_llm_wrapper(model_name=config.routing_llm, **kwargs)
        self.setup()
        
    def setup(self):
        
        self.history = []
        self.display_history = []
        self.sql_history = []
        self.is_routing = False
        
        system_instruction = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/chat/chat.txt'))
# Only answer questions related to finance and accounting.
# If the question is not related to finance and accounting, say You are only allowed to ask questions related to finance and accounting.
        self.history.append(
            {
                'role': 'system',
                'content': system_instruction
            }
        )
        self.text2sql.reset()
        
    def create_new_chat(self, **kwargs):
        self.setup()
    
        
    def routing(self, user_input):
        try:
            
            routing_log = deepcopy(self.display_history)
            routing_instruction = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/chat/routing.txt'))
            
            if len(routing_log) < 1:
                routing_log = []
            
            routing_log.append(
                {
                    'role': 'user',
                    'content': routing_instruction.format(user_input = user_input)
                }
            )
            
            response = self.routing_llm(routing_log)
            routing = get_json_from_text_response(response, new_method=True)['trigger']
            return routing
        
        except Exception as e:
            logging.error(f"Routing error: {e}")
            return False
    
    
    def summarize_and_get_task(self, messages):
        short_messages = messages[-5:] # Get the last 5 messages
        
        task = short_messages[-1]['content']
        short_messages.pop()
        
        system_instruction = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/chat/summarize.txt'))
        
        prompt = [
            {
                'role': 'system',
                'content': system_instruction
            },
            {
                'role': 'user',
                'content': f"Here is the conversation history\n\n{flatten_conversation(short_messages)}.\n\n Here is the current request from user\n\n{task}"
                            
            }
        ]
        
        response = self.routing_llm(prompt)
        return response
        
        
    def _solve_text2sql(self, user_input):
        
        task = user_input
        if self.config.get_task:
            logging.info("Summarizing and getting task")
            task = self.summarize_and_get_task(self.display_history.copy())
        
        table_strings = ""
        
        self.sql_history, error_messages, execution_tables =  self.text2sql.solve(task)
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/sql_history.json', 'w') as file:
            json.dump(self.sql_history, file)
        
        table_strings = utils.table_to_markdown(execution_tables)

        self.history.append(
            {
                'role': 'user',
                'content': f"""
                
            <task>    
            You are provided with the following data:
            
            <table>
            
            {table_strings}
            
            <table>
            
            However, your user cannot see this database. Think step-by-step Analyze and answer the following user question:
            
            <input>
            
            {user_input}
            
            <input>
            
            You should provide the answer based on the provided data. 
            The data often has unclear column names and datetime, but you can assume the data is correct and relevant to the task.
            
            If the provided data is not enough, try your best.
            Answer the question as natural as possible. 
            
            </task>
            """
            }
        )

        
        return table_strings
        
    
    def solve_text2sql(self, user_input):
        return self._solve_text2sql(user_input)
        
        
    def __reasoning(self, user_input, routing = False):
        
        table_strings = ""
        if routing:
            logging.info("Routing triggered")
            table_strings = self.solve_text2sql(user_input)
        
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
        
        self.is_routing = False # Reset the routing flag
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        start = time.time()
        
        # Routing
        self.is_routing = self.routing(user_input)
        if self.is_routing:
            yield '\n\nAnalyzing '
        
            table_strings = self.__reasoning(user_input, self.is_routing)
            end = time.time()
            
            yield 'in {:.2f}s\n\n'.format(end - start)
            yield table_strings + '\n\n'
        
        else:
            table_strings = self.__reasoning(user_input, self.is_routing)
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
            
        self.get_generated_response(''.join(text_response), table_strings)
        
            
        
    def chat(self, user_input):
        
        self.is_routing = False # Reset the routing flag
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        start = time.time()
        
        self.is_routing = self.routing(user_input)
        table_strings = self.__reasoning(user_input, self.is_routing)
        response = self.llm(self.history)
        
        end = time.time()
        logging.info(f"Reasoning time without streaming: {end - start}s")
        
        self.get_generated_response(response, table_strings)
        return table_strings + '\n\n' + response
    
    
    
    def get_generated_response(self, assistant_response, table_strings = ""):
        
        if table_strings == "":
            
            self.display_history.append({
                'role': 'assistant',
                'content': assistant_response
            })
        
        else:
            self.display_history.append({
                'role': 'assistant',
                'content': table_strings + "\n\n" + assistant_response
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
    
    tables: List = [] 
    is_routing: bool = False
    last_sql_id: str = ""
    
    message_saver: BaseSemantic
    
    conversation_id: str = ""
    
    def __init__(self, message_saver: BaseSemantic, **kwargs):
        super().__init__(message_saver= message_saver, **kwargs)
        
        # self.llm = utils.get_llm_wrapper(model_name=config.llm, **kwargs)
        # self.routing_llm = utils.get_llm_wrapper(model_name=config.routing_llm, **kwargs)
        # self.setup()
        
    def save_sql(self, task):
        response = self.sql_history[-1]['content']  
        codes = get_code_from_text_response(response)
        sqls = []
        for code in codes:
            if code['language'] == 'sql':
                sqls.append(code['code'])
                
        self.last_sql_id = self.message_saver.add_sql(self.conversation_id, task, sqls)
        self.sql_history[-1]['sql_id'] = self.last_sql_id
        
        
    def solve_text2sql(self, task):
        table_strings = self._solve_text2sql(task)
        self.save_sql(task)
        return table_strings
        
        
    def create_new_chat(self, user_id: str = "test_user"):
        self.setup()
        self.conversation_id = self.message_saver.create_conversation(user_id)
        
        
    def get_generated_response(self, assistant_response, table_strings = ""):
        
        
            # self.sql_index = len(self.history) - 1
        super().get_generated_response(assistant_response, table_strings)
        
        if self.is_routing: # Previous message triggered the text2sql
            self.history[-1]['sql_id'] = self.last_sql_id
        
        self.message_saver.add_message(self.conversation_id, self.history, self.sql_history)
        
        
    def update_feedback(self, feedback):
        
        score = 0
        if feedback.lower() in {'good', 'like' }:
            score = 1
        elif feedback.lower() in {'bad', 'dislike'}:
            score = -1
            
        self.history[-1]['feedback'] = score
            
        if self.is_routing:
            self.message_saver.sql_feedback(self.last_sql_id, score)
        self.message_saver.add_message(self.conversation_id, self.history, self.sql_history)
            
        
        
    