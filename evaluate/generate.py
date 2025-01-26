import json 
import pandas as pd
import numpy as np

import os
import sys 
sys.path.append('..')

from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    GPT4O_MINI_CONFIG,
    GPT4O_CONFIG,
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG,
    TEXT2SQL_MEDIUM_GEMINI_CONFIG,
    TEXT2SQL_GEMINI_PRO_CONFIG,
    TEXT2SQL_4O_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    HORIZONTAL_PROMPT_BASE,
    HORIZONTAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL,
)
import agent.text2sql_utils as utils



from initialize import initialize_text2sql
from llm.llm_utils import get_json_from_text_response, get_code_from_text_response

from concurrent.futures import ThreadPoolExecutor, as_completed


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def append_json_to_file(json_obj, file_path):
    with open(file_path, 'a') as f:
        json.dump(json_obj, f)
        f.write('\n')

def single_solver(text2sql_config, prompt_config, batch_questions, using_cache=False, file_path=None):
    """
    Run a single solver on a batch of questions
    """

    solver = initialize_text2sql(text2sql_config, prompt_config)

    responses = []
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

        if file_path:
            append_json_to_file({
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

    for question in questions:
        if version and version != 'all':
            if question['version'] != version:
                continue
        batch_question.append(question)
        if len(batch_question) == batch_size:
            batch_questions.append(batch_question)
            batch_question = []
    
    # Last batch
    if batch_question:
        batch_questions.append(batch_question)

    # For testing
    batch_questions = batch_questions[:4]

    if version:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, f"../data/{text2sql_config.get('llm', 'unknown')}__{version}.jsonl")
    else:
        file_path = f"../data/{text2sql_config.get('llm', 'unknown')}_all.jsonl"

    results = []

    if multi_thread:
        print("Using multi-threading")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_solver, text2sql_config, prompt_config, batch_question, using_cache, file_path) for batch_question in batch_questions]

            for future in as_completed(futures):
                results.extend(future.result())
    else:
        print("Using single-threading")
        for batch_question in batch_questions:
            results.extend(single_solver(text2sql_config, prompt_config, batch_question, using_cache, file_path))

    return results


def get_text2sql_config(llm_name):
    if 'gemini' in llm_name:
        if llm_name == 'gemini-flash':
            return TEXT2SQL_GEMINI_PRO_CONFIG
        return TEXT2SQL_MEDIUM_GEMINI_CONFIG
    
    if 'gpt-4o' in llm_name:
        if 'mini' not in llm_name:
            return TEXT2SQL_4O_CONFIG
        return TEXT2SQL_FAST_OPENAI_CONFIG

    if 'deepseek' in llm_name:
        return TEXT2SQL_DEEPSEEK_V3_CONFIG

    else:
        config = TEXT2SQL_MEDIUM_GEMINI_CONFIG
        config['sql_llm'] = llm_name    
        return config

def solve(args):
    text2sql_config = get_text2sql_config(args.llm)
    prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL
    version = args.version

    with open('../data/generated_questions.json') as f:
        questions = json.load(f)
        print(len(questions))

    results = _solve(text2sql_config, prompt_config, questions, using_cache=args.using_cache, version=version, batch_size=args.batch_size, max_workers=args.max_workers, multi_thread=args.multi_thread)
    with open('../data/generated_questions_sql.jsonl', 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')


import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='generate_sql', help='task to evaluate')
    parser.add_argument('--version', default='v0', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_workers', default=4, type=int)
    parser.add_argument('--multi_thread', default=False, type=bool)
    parser.add_argument('--using_cache', default=False, type=bool)
    parser.add_argument('--llm', default='gpt-4o-mini', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.task == 'generate_sql':
        solve(args)

    
    
        
