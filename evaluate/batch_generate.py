from utils import append_jsonl_to_file, get_available_path
import random

import datetime

import sys
sys.path.append('..')

from initialize import initialize_text2sql

def prepare_messages(text2sql_config, prompt_config, questions, output_path = None, enhance = None):

    text2sql_config['sql_example_top_k'] = random.randint(2, 3)
    text2sql_config['account_top_k'] = random.randint(4, 6)
    adjust = random.randint(0, 1)

    task = questions.get('question')
    ids = questions.get('ids')

    solver = initialize_text2sql(text2sql_config, prompt_config)

    temp_message = solver.get_solver_template_message(task, adjust = adjust, enhance = enhance)

    msg_obj = {
        'date' : datetime.datetime.now().strftime("%Y-%d-%m %H:%M:%S"),
        'ids': ids,
        'message': temp_message
    }

    if output_path:
        append_jsonl_to_file(output_path, msg_obj)
    
    return msg_obj

