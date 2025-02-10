from utils import (
    append_jsonl_to_file, 
    get_available_path, 
    get_text2sql_config, 
    get_prompt_config, 
    get_avaliable_questions
    )

import random
import os
import json

import datetime

import sys
sys.path.append('..')

from initialize import initialize_text2sql
from concurrent.futures import ThreadPoolExecutor, as_completed


def prepare_messages(text2sql_config, prompt_config, questions, output_path = None, enhance = None):

    text2sql_config['sql_example_top_k'] = random.randint(2, 3)
    text2sql_config['account_top_k'] = random.randint(4, 6)
    adjust = random.randint(0, 1)

    task = questions.get('question')
    ids = questions.get('ids')

    solver = initialize_text2sql(text2sql_config, prompt_config)

    temp_message = solver.get_solver_template_message(task, adjust_table = adjust, enhance = enhance)

    msg_obj = {
        'date' : datetime.datetime.now().strftime("%Y-%d-%m %H:%M:%S"),
        'ids': ids,
        'message': temp_message
    }

    if output_path:
        append_jsonl_to_file(msg_obj, output_path)
    
    return msg_obj



def prepare_messages_template(text2sql_config, prompt_config, input_path, output_path, reference_path = None, enhance = None, multi_thread = False, max_workers = 10):

    questions = get_avaliable_questions(input_path, [output_path, reference_path])[:2500]

    # Test
    # questions = questions[:10]

    print(f"Total questions: {len(questions)}")
    
    if multi_thread:
        print("Using multi-threading")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(prepare_messages, text2sql_config, prompt_config, question, output_path, enhance) for question in questions]
            results = [future.result() for future in as_completed(futures)]
    else:
        print("Using single-threading")
        results = [prepare_messages(text2sql_config, prompt_config, question, output_path, enhance) for question in questions]
    
    return results


