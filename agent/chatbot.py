from .base import BaseAgent

from .const import ChatConfig
from . import text2sql_utils as utils
from .text2sql import Text2SQL, Text2SQLOutput
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
    solver_ouputs: dict = dict()
    solver_ids: List[str] = []
    
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

        display_instruction = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/chat/display.txt'))

        self.display_history.append(
            {
                'role': 'system',
                'content': display_instruction
            }
        )

        

        self.text2sql.reset()
        self.solver_ids = []
        self.solver_ouputs = dict()
        
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
            return {
                'trigger': routing
            }
        
        except Exception as e:
            logging.error(f"Routing error: {e}")
            return {
                'trigger': False
            }
    
    
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
    

    def routing_v2(self, user_input):
        try:
            routing_log = deepcopy(self.display_history)
            routing_instruction = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/chat/routing_v2.txt'))
            
            routing_log.append(
                {
                    'role': 'user',
                    'content': routing_instruction.format(user_input = user_input)
                }
            )
            
            response = self.routing_llm(routing_log)
            print("ROUTING RESPONSE")
            print(response)

            # Beautiful response
            dict_response = utils.get_content_with_heading_tag(response, tag="###")
            print(dict_response)

            # ugly response
            if dict_response.get('decision', None) is None:
                words = response.split("\n")

                trigger = words[0]
                task = "\n".join(words[1:]).strip()

                dict_response = {
                    'decision': trigger,
                    'task': task
                }

            trigger = dict_response.get('decision', 'False').lower().replace("{",'').replace("}",'').strip()

            if trigger in {'true', 'yes', '1', 'y', 't'}:
                trigger = True
            else:
                trigger = False

            return {
                'trigger': trigger,
                'task': dict_response.get('task', None)
            }
            
        except Exception as e:
            logging.error(f"Routing error: {e}")
            return {
                'trigger': False,
                'task': None
            }
        

    def _solve_text2sql_v2(self, user_input, routing_log, **kwargs):
        
        task = routing_log.get('task', user_input)

        return self._solver(user_input, task, **kwargs)
    

    def _solve_text2sql(self, user_input, **kwargs):
        """ 
            Summarize and get task
        """
        
        task = user_input
        if self.config.get_task:
            logging.info("Summarizing and getting task")
            task = self.summarize_and_get_task(self.display_history.copy())
        
        return self._solver(user_input, task, **kwargs)


        
    def _solver(self, user_input, task, **kwargs):
        
        table_strings = ""
        
        output =  self.text2sql.solve(task, **kwargs)
        self.sql_history = output.history
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/sql_history.json', 'w') as file:
            json.dump(self.sql_history, file)
        
        table_strings = utils.table_to_markdown(output.execution_tables)

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

        
        return output
        
    
    def solve_text2sql(self, user_input, version = "v1", **kwargs):
        if version == "v2":
            return self._solve_text2sql_v2(user_input, **kwargs)
        return self._solve_text2sql(user_input, **kwargs)
        
        
    def __reasoning(self, user_input, routing_log: dict, version = "v1", **kwargs):
        
        routing = routing_log.get('trigger', False)

        table_strings = ""
        if routing:
            logging.info("Routing triggered")
            output = self.solve_text2sql(user_input, version = version, routing_log = routing_log, **kwargs)
            table_strings = utils.table_to_markdown(output.execution_tables)
        
        else:
            logging.info("Routing not triggered")
            self.history.append(
                {
                    'role': 'user',
                    'content': user_input
                }
            )
        print(table_strings)
        return table_strings
        
        # return response
        
    def stream(self, user_input, version = "v1", **kwargs):
        
        self.is_routing = False # Reset the routing flag
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        start = time.time()
        
        # Routing
        if version == "v2":
            routing_log = self.routing_v2(user_input)
            self.is_routing = routing_log['trigger']
        else:
            routing_log = self.routing(user_input)
            self.is_routing = routing_log['trigger']

        if self.is_routing:
            yield '\n\nAnalyzing '
        
            table_strings = self.__reasoning(user_input, version=version, routing_log=routing_log, **kwargs)
            end = time.time()
            
            yield 'in {:.2f}s\n\n'.format(end - start)
            yield table_strings + '\n\n'
        
        else:
            table_strings = self.__reasoning(user_input, version=version, routing_log=routing_log, **kwargs)
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
        
            
        
    def chat(self, user_input, version = "v1", **kwargs):
        
        self.is_routing = False # Reset the routing flag
        
        self.display_history.append({
            'role': 'user',
            'content': user_input
        })
        
        start = time.time()
        
        if version == "v2":
            routing_log = self.routing_v2(user_input)
            self.is_routing = routing_log['trigger']
        else:
            routing_log = self.routing(user_input)
            self.is_routing = routing_log['trigger']

        table_strings = self.__reasoning(user_input, routing=self.is_routing, version=version, **kwargs)
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
        
    message_saver: BaseSemantic
    conversation_id: str = None
        
    def save_sql(self, output: Text2SQLOutput):
        print("Saving SQL")
        dict_output = output.convert_to_dict()

        # Save the single mesage
        self.message_saver.add_solver_output(dict_output)

        solver_id = dict_output['solver_id']

        # Add new solver id to the list
        if solver_id not in self.solver_ouputs:
            self.solver_ouputs[solver_id] = []
            self.solver_ids.append(solver_id)

        self.solver_ouputs[solver_id].append(len(self.history) - 1)
        
        
    def solve_text2sql(self, task, version = 'v1', **kwargs):
        output = super().solve_text2sql(task, version = version, **kwargs)
        
        self.save_sql(output)
        return output
        
        
    def create_new_chat(self, user_id: str = "test_user"):
        self.setup()
        self.conversation_id = self.message_saver.create_conversation(user_id)
        
        
    def get_generated_response(self, assistant_response, table_strings = ""):
        
        
            # self.sql_index = len(self.history) - 1
        super().get_generated_response(assistant_response, table_strings)
        
        if self.is_routing: # Previous message triggered the text2sql
            
            # Save the last sql_id
            self.history[-1]['sql_id'] = self.solver_ouputs[self.solver_ids[-1]][-1]
        
        self.message_saver.add_message(self.conversation_id, self.history, self.solver_ids)
        
        
    def update_feedback(self, feedback):
        
        score = 0
        if feedback.lower() in {'good', 'like' }:
            score = 1
        elif feedback.lower() in {'bad', 'dislike'}:
            score = -1
            
        self.history[-1]['feedback'] = score

        last_solver_id = self.solver_ids[-1]
        last_output_id = self.solver_ouputs[last_solver_id][-1]
            
        if self.is_routing:
            self.message_saver.sql_feedback(last_solver_id, last_output_id, score)
        self.message_saver.add_message(self.conversation_id, self.history, self.solver_ids)
            
        
        
    