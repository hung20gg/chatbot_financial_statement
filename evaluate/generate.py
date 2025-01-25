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
    TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG
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
        json.dump(json_obj, f, indent=4)
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

    for i, code in enumerate(codes):
        responses.append({
            'id': ids[i],
            'question': prompt,
            'table': table_str,
            'sql': code
        })

        if file_path:
            append_json_to_file({
                'id': ids[i],
                'question': prompt,
                'table': table_str,
                'sql': code
            }, file_path)


    return responses

def solve(text2sql_config, prompt_config, questions, using_cache=False, version = None, batch_size=5, max_workers=4, multi_thread=False):
    """
    Run a single solver on a batch of questions in parallel
    """
    batch_questions = []
    batch_question = []

    for question in questions:
        print(question)
        if version:
            if question['version'] != version:
                continue
        batch_question.append(question)
        if len(batch_question) == batch_size:
            batch_questions.append(batch_question)
            batch_question = []
    
    # Last batch
    if batch_question:
        batch_questions.append(batch_question)

    if version:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, f"../data/{text2sql_config.get('sql_llm', 'unknown')}__{version}.jsonl")
    else:
        file_path = f"../data/{text2sql_config.get('sql_llm', 'unknown')}_all.jsonl"

    results = []

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_solver, text2sql_config, prompt_config, batch_question, using_cache, file_path) for batch_question in batch_questions]

            for future in as_completed(futures):
                results.extend(future.result())
    else:
        for batch_question in batch_questions:
            results.extend(single_solver(text2sql_config, prompt_config, batch_question, using_cache, file_path))

    return results

def main():
    text2sql_config = TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG
    prompt_config = VERTICAL_PROMPT_UNIVERSAL
    version = 'v1'

    with open('../data/generated_questions.json') as f:
        questions = json.load(f)
        print(len(questions))

    # Test    
    questions = questions[:10]

    results = solve(text2sql_config, prompt_config, questions, using_cache=False, version=version, batch_size=5, max_workers=4)
    with open('../data/generated_questions_sql.jsonl', 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')

if __name__ == '__main__':
    main()

    
    
        
