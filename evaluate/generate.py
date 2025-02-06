import json 
import pandas as pd
import numpy as np

import random

import os
import sys 

from utils import append_jsonl_to_file, get_available_path
sys.path.append('..')

from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    GPT4O_MINI_CONFIG,
    GPT4O_CONFIG,
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG,
    TEXT2SQL_MEDIUM_GEMINI_CONFIG,
    TEXT2SQL_GEMINI_PRO_CONFIG,
    TEXT2SQL_THINKING_GEMINI_CONFIG,
    TEXT2SQL_4O_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    HORIZONTAL_PROMPT_BASE,
    HORIZONTAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI,
)
import agent.text2sql_utils as utils



from initialize import initialize_text2sql
from llm.llm_utils import get_json_from_text_response, get_code_from_text_response

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_QUESTION = 1000

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def get_text2sql_config(llm_name):
    if 'gemini' in llm_name:

        if 'thinking' in llm_name:
            return TEXT2SQL_THINKING_GEMINI_CONFIG
        if 'gemini-pro' in llm_name:
            return TEXT2SQL_GEMINI_PRO_CONFIG
        return TEXT2SQL_FAST_GEMINI_CONFIG
    
    if 'gpt-4o' in llm_name:
        if 'mini' not in llm_name:
            return TEXT2SQL_4O_CONFIG
        return TEXT2SQL_FAST_OPENAI_CONFIG

    if 'deepseek-chat' in llm_name:
        return TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG

    else:
        config = TEXT2SQL_FAST_GEMINI_CONFIG
        config['sql_llm'] = llm_name    
        return config



## ============ SQL SOLVER ============ ##

def single_solver(text2sql_config, prompt_config, batch_questions, using_cache=False, file_path=None):
    """
    Run a single solver on a batch of questions
    """

    # Initialize the solver
    solver = initialize_text2sql(text2sql_config, prompt_config)

    responses = []

    # Loop through the questions
    for question in batch_questions:
        prompt = question['question']
        ids = question['ids']
        if not using_cache:
            solver.reset()
        his, err, tables = solver.solve(prompt)
        table_str = utils.table_to_markdown(tables)

        # Get the SQL code from the last response
        codes = get_code_from_text_response(his[-1]['content'])

        sql = []
        for code in codes:
            if code.get('language') == 'sql':
                sql.append(code.get('code',''))

        responses.append({
            'ids': ids,
            'question': prompt,
            'table': table_str,
            'reasoning': his[-1]['content'],
            'sql': sql
        })

        # Save to file
        if file_path:
            append_jsonl_to_file({
                'ids': ids,
                'question': prompt,
                'table': table_str,
                'reasoning': his[-1]['content'],
                'sql': sql
            }, file_path)


    return responses


def _solve(text2sql_config, prompt_config, questions, using_cache=False, version = None, batch_size=5, max_workers=4, multi_thread=False):
    """
    Run a single solver on a batch of questions in parallel
    """
    batch_questions = []
    batch_question = []

    sql_llm = text2sql_config.get('sql_llm', 'unknown').replace('/', '__')

    print(f'==== Version: {version} ====')

    count = 0
    for question in questions:
        if 'version' in question:
            if version and question['version'] != version: # Specific version v1 v2 v3
                continue

            # Else is all or specific version
        batch_question.append(question)
        if len(batch_question) == batch_size:
            batch_questions.append(batch_question)
            batch_question = []
        count += 1
        if count >= MAX_QUESTION:
            break
    
    # Last batch
    if batch_question:
        batch_questions.append(batch_question)


    # Get the file path
    if version:
        current_dir = os.path.dirname(__file__)
        output_path = os.path.join(current_dir, f"../data/{sql_llm}__{version}.jsonl")
    else:
        output_path = f"../data/{sql_llm}_all.jsonl"

    # output_path = get_available_path(output_path)

    print(f"==== Total questions: {len(questions)} =====")
    print(f'==== Saving to: {output_path} =====')

    # Run the solver
    results = []

    if multi_thread:
        print("Using multi-threading")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_solver, text2sql_config, prompt_config, batch_question, using_cache, output_path) for batch_question in batch_questions]

            for future in as_completed(futures):
                results.extend(future.result())
    else:
        print("Using single-threading")
        for batch_question in batch_questions:
            results.append(single_solver(text2sql_config, prompt_config, batch_question, using_cache, output_path))

    return results



## ============ FAKE MESSAGES ============ ##

def single_fake_messages(text2sql_config,  batch_questions, using_cache=False, file_path=None):
    
    text2sql_config['sql_example_top_k'] = random.randint(1, 6) // 3 # 0,0, 1, 1, 1, 2

    random_config = random.choice([FIIN_VERTICAL_PROMPT_UNIVERSAL, FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY, FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY, FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI])
    random_table = random.randint(0, 1)

    solver = initialize_text2sql(text2sql_config, random_config)
    global_history = []
    for question in batch_questions:
        prompt = question['question']
        reasoning = question['reasoning']

        if not using_cache:
            solver.reset()
        his, err, tables = solver.solve(prompt, inject_reasoning=reasoning, adjust_table=random_table) # Fake reasoning

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
            results.append(single_fake_messages(text2sql_config, prompt_config, batch_question, using_cache, file_path))


    return results

# ============ GENERATE FAKE MESSAGE RUNNER ============

def generate_fake_messages(args):
    text2sql_config = get_text2sql_config(args.llm)

    version = args.version

    selected_questions = []
    with open(args.path) as f: # JSONL
        for line in f:
            question = json.loads(line)
            selected_questions.append(question)
        
        print(f"Total questions: {len(selected_questions)}")
    
    base_name = os.path.basename(args.path).replace('.jsonl', '')

    if version:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, f"../data/message_{base_name}.jsonl")
    else:
        file_path = os.path.join(current_dir, f"../data/message_{base_name}.jsonl")

    file_path = get_available_path(file_path)

    results = _fake_solve(text2sql_config, selected_questions, using_cache=args.using_cache, file_path=file_path, max_workers=args.max_workers, multi_thread=args.multi_thread)


# ============ GENERATE SQL RUNNER ============

def generate_sql(args):
    text2sql_config = get_text2sql_config(args.llm)

    # Change the template here
    if 'openai' in args.template:
        print("===== Using openai =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI
    elif 'simplify' in args.template:
        print("===== Using simplify =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY
    else:
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL

    
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

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                done_ids.add(data['ids'])


    selected_questions = []

    
    # Load the questions
    if file_path.endswith('.json'):

        with open(file_path) as f:
            questions = json.load(f)
            for question in questions:
                if question['ids'] not in done_ids:
                    selected_questions.append(question)

    elif file_path.endswith('.jsonl'):
        with open(file_path) as f:
            for line in f:
                question = json.loads(line)
                if question['ids'] not in done_ids:
                    selected_questions.append(question)
    
    else:
        raise ValueError("Invalid file path")
        
    print(f"Total questions: {len(selected_questions)}")

    # Run the solver
    results = _solve(text2sql_config, prompt_config, selected_questions, using_cache=args.using_cache, version=version, batch_size=args.batch_size, max_workers=args.max_workers, multi_thread=args.multi_thread)



import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='generate_sql', help='task to evaluate')
    parser.add_argument('--version', default='v0', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_workers', default=4, type=int)
    parser.add_argument('--multi_thread', default=False, type=bool)
    parser.add_argument('--using_cache', default=True, type=bool)
    parser.add_argument('--path', default='../data/deepseek-chat__v0_good.jsonl', type=str)
    parser.add_argument('--llm', default='gpt-4o-mini', type=str)
    parser.add_argument('--template', default='vertical', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.task == 'generate_sql':
        generate_sql(args)

    elif args.task == 'generate_messages':
        generate_fake_messages(args)

    
    
        
