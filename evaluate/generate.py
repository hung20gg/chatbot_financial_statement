import json 
import pandas as pd
import numpy as np
import time
import datetime

import random

import os
import sys 

from utils import (
    append_jsonl_to_file, 
    get_available_path, 
    get_text2sql_config, 
    get_prompt_config, 
    get_avaliable_questions
    )

sys.path.append('..')


import agent.text2sql_utils as utils


from batch_generate import prepare_messages_template
from initialize import initialize_text2sql
from llm.llm_utils import get_json_from_text_response, get_code_from_text_response

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_QUESTION = 2000
DB_VERSION = 'v3'

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def single_ner(text2sql_config, prompt_config, batch_questions, using_cache=False, enhance = None, file_path=None, rotate_key = False):
    """
    Run a single ner on a batch of questions
    """

    # Initialize the solver
    solver = initialize_text2sql(text2sql_config, prompt_config, version=DB_VERSION, rotate_key=rotate_key)
    is_exp_model = 'exp' in text2sql_config['sql_llm']

    responses = []

    # Loop through the questions
    for question in batch_questions:

        start = time.time()

        prompt = question['question']
        ids = question['ids']
        if not using_cache:
            solver.reset()

        ner_messages = solver._llm_get_stock_code_and_suitable_row(prompt)

        # Get the SQL code from the last response

        end = time.time()

        answer = {
            'ids': ids,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': prompt,
            'duration': end - start,
            'ner': ner_messages
        }

        responses.append(answer)

        # Save to file
        if file_path:
            append_jsonl_to_file(answer, file_path)
        
        # if is_exp_model:
        #     time.sleep(5)


    return responses



## ============ SQL SOLVER ============ ##

def single_solver(text2sql_config, prompt_config, batch_questions, using_cache=False, enhance = None, file_path=None):
    """
    Run a single solver on a batch of questions
    """

    text2sql_config['sql_example_top_k'] = random.randint(2, 3)
    text2sql_config['account_top_k'] = random.randint(4, 6)

    # Initialize the solver
    solver = initialize_text2sql(text2sql_config, prompt_config, version=DB_VERSION,)
    is_exp_model = 'exp' in text2sql_config['sql_llm']

    responses = []

    # Loop through the questions
    for question in batch_questions:

        start = time.time()

        prompt = question['question']
        ids = question['ids']
        if not using_cache:
            solver.reset()

        # ner_messages = solver._llm_get_stock_code_and_suitable_row(prompt)
        output = solver.solve(prompt, enhance=enhance)

        his, err, tables = output.history, output.error_messages, output.execution_tables
        ner_messages = output.extraction_msg
        
        table_str = utils.table_to_markdown(tables)

        # Get the SQL code from the last response
        codes = get_code_from_text_response(his[-1]['content'])

        sql = []
        for code in codes:
            if code.get('language') == 'sql':
                sql.append(code.get('code',''))

        end = time.time()

        answer = {
            'ids': ids,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': prompt,
            'table': table_str,
            'reasoning': his[-1]['content'],
            'sql': sql,
            'messages': his,
            'duration': end - start,
            'ner': ner_messages
        }

        responses.append(answer)

        # Save to file
        if file_path:
            append_jsonl_to_file(answer, file_path)
        
        if is_exp_model:
            time.sleep(8)


    return responses


def _solve(text2sql_config, prompt_config, questions, using_cache=False, version = None, enhance = None, batch_size=5, max_workers=4, multi_thread=False, task = 'sql', rotate_key = False):
    """
    Run a single solver on a batch of questions in parallel
    """
    batch_questions = []
    batch_question = []

    # If enhance, batch size is 1
    if enhance:
        batch_size = 1

    sql_llm = text2sql_config.get('sql_llm', 'unknown').replace('/', '__')

    print(f'==== Version: {version} ====')

    count = 0
    for question in questions:
        if 'version' in question:
            if version and question['version'] != version: # Specific version v1 v2 v3
                continue

            # Else is all or specific version
        batch_question.append(question)
        count += 1

        if len(batch_question) == batch_size:
            batch_questions.append(batch_question)
            batch_question = []
        
        if count >= MAX_QUESTION:
            break
    
    # Last batch
    if batch_question:
        batch_questions.append(batch_question)


    # Get the file path
    if task == 'sql':
        if version:
            current_dir = os.path.dirname(__file__)
            output_path = os.path.join(current_dir, f"../data/{sql_llm}__{version}.jsonl")
        else:
            output_path = f"../data/{sql_llm}_all.jsonl"
    else:
        if version:
            current_dir = os.path.dirname(__file__)
            output_path = os.path.join(current_dir, f"../data/ner_{sql_llm}__{version}.jsonl")
        else:
            output_path = f"../data/ner_{sql_llm}_all.jsonl"

    # output_path = get_available_path(output_path)

    print(f"==== Total questions: {count} =====")
    print(f"==== Total batch: {len(batch_questions)} =====")
    print(f'==== Saving to: {output_path} =====')

    # Run the solver
    results = []

    if multi_thread:
        print("Using multi-threading")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if task == 'sql':
                futures = [executor.submit(single_solver, text2sql_config, prompt_config, batch_question, using_cache, enhance, output_path) for batch_question in batch_questions]
            else:
                futures = [executor.submit(single_ner, text2sql_config, prompt_config, batch_question, using_cache, enhance, output_path) for batch_question in batch_questions]

            for future in as_completed(futures):
                results.extend(future.result())
    else:
        print("Using single-threading")
        for batch_question in batch_questions:
            if task == 'sql':
                results.append(single_solver(text2sql_config, prompt_config, batch_question, using_cache, enhance, output_path))
            else:
                results.append(single_ner(text2sql_config, prompt_config, batch_question, using_cache, enhance, output_path))
    return results



## ============ FAKE MESSAGES ============ ##

def single_fake_messages(text2sql_config,  batch_questions, using_cache=False, file_path=None):
    
    text2sql_config['sql_example_top_k'] = random.randint(1, 6) // 3 # 0, 0, 1, 1, 1, 2

    random_config = random.choice(['vertical', 'openai', 'openai', 'simpify', 'simpify', 'openai'])
    random_table = random.randint(0, 1)

    prompt_config = get_prompt_config(random_config)

    solver = initialize_text2sql(text2sql_config, prompt_config, version=DB_VERSION)
    global_history = []
    for question in batch_questions:
        prompt = question['question']
        reasoning = question['reasoning']

        if not using_cache:
            solver.reset()

        output = solver.solve(prompt, inject_reasoning=reasoning, adjust_table=random_table) # Fake reasoning
        his, err, tables = output.history, output.error_messages, output.execution_tables
        
        global_history = his
    
    if file_path:
        append_jsonl_to_file(global_history, file_path)
    return global_history



def _fake_solve(text2sql_config, questions, using_cache=False, file_path = None, max_workers=4, multi_thread=False):

    batch_questions = []
    index = 0

    print(f"Total questions: {len(questions)}")
    bs = 0

    while index < len(questions):
        batch_size = max(random.randint(1, 6)//3, 1) # 1,1,1,1,1,2
        batch_question = questions[index:index+batch_size]
        index += batch_size
        batch_questions.append(batch_question)
        bs += len(batch_question)

    print(f"Total batch: {len(batch_questions)}")
    print(f"Total questions: {bs}")

    results = []

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_fake_messages, text2sql_config, batch_question, using_cache, file_path) for batch_question in batch_questions]

            for future in as_completed(futures):
                results.extend(future.result())
    else:
        for batch_question in batch_questions:
            results.append(single_fake_messages(text2sql_config, batch_question, using_cache, file_path))


    return results

# ============ GENERATE FAKE MESSAGE RUNNER ============

def generate_fake_messages(args):
    text2sql_config = get_text2sql_config(args.llm)

    version = args.version

    
    
    base_name = os.path.basename(args.path).replace('.jsonl', '')

    if version:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, f"../data/message_{base_name}.jsonl")
    else:
        file_path = os.path.join(current_dir, f"../data/message_{base_name}.jsonl")


    selected_questions = get_avaliable_questions(args.path, file_path)
        
    print(f"Total questions: {len(selected_questions)}")

    # file_path = get_available_path(file_path)

    results = _fake_solve(text2sql_config, selected_questions, using_cache=args.using_cache, file_path=file_path, max_workers=args.max_workers, multi_thread=args.multi_thread)


# ============ GENERATE SQL RUNNER ============

def generate_sql(args):
    text2sql_config = get_text2sql_config(args.llm)

    # Change the template here
    prompt_config = get_prompt_config(args.template)

    
    # Get the version
    version = args.version
    sql_llm = text2sql_config.get('sql_llm', 'unknown').replace('/', '__')

    if args.path:
        file_path = args.path
    else:
        file_path = '../data/generated_questions.json'
    

    # Check if the file exists and if the question is already solved
    if version:
        current_dir = os.path.dirname(__file__)
        output_path = os.path.join(current_dir, f"../data/{sql_llm}__{version}.jsonl")
    else:
        output_path = f"../data/{sql_llm}_all.jsonl"

    # output_path = get_available_path(output_path)

    selected_questions = get_avaliable_questions(file_path, output_path)

    if args.rotate_api:
        print("Using rotate API")
        rotate_key = True

        
    print(f"Total questions: {len(selected_questions)}")

    # Run the solver
    results = _solve(text2sql_config, prompt_config, selected_questions, using_cache=args.using_cache, version=version, enhance=args.enhance, batch_size=args.batch_size, max_workers=args.max_workers, multi_thread=args.multi_thread, rotate_key=args.rotate_api)




def generate_ner(args):
    text2sql_config = get_text2sql_config(args.llm)

    # Change the template here
    prompt_config = get_prompt_config(args.template)

    
    # Get the version
    version = args.version
    sql_llm = text2sql_config.get('sql_llm', 'unknown').replace('/', '__')

    if args.path:
        file_path = args.path
    else:
        file_path = '../data/generated_questions.json'
    

    # Check if the file exists and if the question is already solved
    if version:
        current_dir = os.path.dirname(__file__)
        output_path = os.path.join(current_dir, f"../data/{sql_llm}__{version}.jsonl")
    else:
        output_path = f"../data/{sql_llm}_all.jsonl"

    # output_path = get_available_path(output_path)

    selected_questions = get_avaliable_questions(file_path, output_path)

        
    print(f"Total questions: {len(selected_questions)}")

    # Run the solver
    results = _solve(text2sql_config, prompt_config, selected_questions, using_cache=args.using_cache, version=version, enhance=args.enhance, batch_size=args.batch_size, max_workers=args.max_workers, multi_thread=args.multi_thread, task='ner')


# ============ GENERATE SQL TEMPLATE ============

def generate_sql_template(args):
    text2sql_config = get_text2sql_config(args.llm)

    # Change the template here
    prompt_config = get_prompt_config(args.template)

    
    # Get the version
    version = args.version
    llm = text2sql_config.get('llm', 'unknown').replace('/', '__')

    if args.path:
        file_path = args.path
    else:
        file_path = '../data/generated_questions.json'
    

    # Check if the file exists and if the question is already solved
    if version:
        current_dir = os.path.dirname(__file__)
        output_path = os.path.join(current_dir, f"../data/template_{llm}__{version}.jsonl")
    else:
        output_path = f"../data/template_{llm}_all.jsonl"

    # output_path = get_available_path(output_path)

    return prepare_messages_template(text2sql_config, prompt_config, file_path, output_path, reference_path=args.reference_path, enhance=args.enhance, multi_thread=args.multi_thread, max_workers=args.max_workers)


import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='generate_sql', help='task to evaluate')
    parser.add_argument('--version', default='v0', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_workers', default=4, type=int)
    parser.add_argument('--multi_thread', default=False, type=bool)
    parser.add_argument('--using_cache', default=True, type=bool)
    parser.add_argument('--path', default='../data/deepseek-chat__v0_good.jsonl', type=str)
    parser.add_argument('--llm', default='gpt-4o-mini', type=str)
    parser.add_argument('--template', default='vertical', type=str)
    parser.add_argument('--enhance', default=None, type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--rotate_api', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.task == 'generate_sql':
        generate_sql(args)

    elif args.task == 'generate_messages':
        generate_fake_messages(args)
    
    elif args.task == 'generate_sql_template':
        generate_sql_template(args)

    elif args.task == 'generate_ner':
        generate_ner(args)
    
    
        
