from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 
sys.path.append('..')
import pandas as pd
import numpy as np
import random
import json
import re
import uuid

from agent.text2sql_utils import get_llm_wrapper
from llm.llm_utils import get_json_from_text_response
import random

import os 
from dotenv import load_dotenv
load_dotenv()


mcq_styles = [
    "Pick the false option among the choices",
    "Pick the true option among the choices",
    "Pick the true option among the choices",
    "Pick the most relevant option among the choices",
]


def append_json_to_file(json_obj, file_path):
    with open(file_path, 'a') as f:
        json.dump(json_obj, f)
        f.write('\n')


def _generate_mcq(llm, data, mcq_style, file_path='tmp.jsonl'):
    """
    Generate MCQ from a question and a table
    """
    system_prompt = f"""
    You are an auditor, and you are tasked to giving multiple choice question to test the knowledge of your colleague. 
    Each question should have 4 choices, and only one correct answer.
    You will be given a general task in <task> tag, and reference tables in <table> tag.
    Your task is to generate multiple choice question based on the task and the table.

    Your question should be precise and clear, and make sure the question can only be answered by the information in the table.
    You should also provide the correct answer and explanation for the correct answer.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
    You will have the following data:

    <task>
    {data['question']}
    </task>

    <table>
    {data['table']}
    </table>

    Your task is to generate multiple choice question based on the task and the table.

    The question might follow the style which {mcq_style}, or you can generate your own style if necessary.
    
    After generating the question, think step by step to answer it.
    """
        },
    ]

    # Generate question
    response = llm(messages)

    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    messages.append(
        {
            "role": "user",
            "content": """
    Based on the multiple choice question, store it into JSON format:

    Assume the "choice 2" is the correct answer. 

    ```json
    {
        "question": "multiple choice question",
        "choices": [
            "choice 1", 
            "choice 2",
            "choice 3",
            "choice 4"
        ],
        "answer": 1,
        "explanation": "explanation"
    }
    ```
    """
        }
    )

    # Get the question in JSON format
    response = llm(messages)

    mcq = get_json_from_text_response(response, new_method=True)

    data['mcq'] = mcq

    append_json_to_file(data, file_path)


def generate_mcq(llm, data, file_path='tmp.jsonl', max_workers=10, multi_thread=False):
    print("Generating MCQ...")
    print("Number of questions:", len(data))

    

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(len(data)):
                future = executor.submit(_generate_mcq, llm, data[i], mcq_styles[random.randint(0, len(mcq_styles)-1)], file_path)
                futures.append(future)
            for future in as_completed(futures):
                future.result()

    else:
        for i in range(len(data)):
            _generate_mcq(llm, data[i], mcq_styles[random.randint(0, len(mcq_styles)-1)], file_path)


def generate():
    llm = get_llm_wrapper(model_name="deepseek-chat")
    data = []

    with open('../data/deepseek-chat__v0.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    max_workers = min(4, len(data))

    generate_mcq(llm, data, file_path='../data/tmp_mcq.jsonl', max_workers=max_workers, multi_thread=True)

if __name__ == "__main__":
    generate()