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


def append_jsonl_to_file(json_obj, file_path):
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
    If you think the provided table is not enough to generate a question, refuse the task and ask for more information.
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

    Assume the "choice 2" is the correct answer. The JSON format should be:
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

    If no question is provided, return an empty JSON object.
    """
        }
    )

    # Get the question in JSON format
    response = llm(messages)

    
    mcq = get_json_from_text_response(response, new_method=True)

    try:
    # Store the question
        data['mcq'] = mcq
        question = dict()
        question['ids'] = data['ids']
        question['question'] = mcq['question']
        question['choices'] = mcq['choices']
        question['answer'] = mcq['answer']
        question['explanation'] = mcq['explanation']

        append_jsonl_to_file(question, file_path)
    except:
        print("Error in generating MCQ =====")
        return

def generate_mcq(llm, input_path, max_workers=10, multi_thread=False):
    llm = get_llm_wrapper(llm)
    print("Generating MCQ...")
    
    
    basename = os.path.basename(input_path)
    output_path = f"../data/mcq_{basename}"

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                done_ids.add(data['ids'])

    data = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                line = json.loads(line)
                if line['ids'] in done_ids:
                    continue
                data.append(line)
            except:
                continue

    print("Number of questions:", len(data))


    global mcq_styles

    results = []
    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_mcq, llm, data[i], mcq_styles[random.randint(0, len(mcq_styles)-1)], output_path) for i in range(len(data))]
            
            for future in as_completed(futures):
                results.extend(future.result())

            

    else:
        for i in range(len(data)):
            _generate_mcq(llm, data[i], mcq_styles[random.randint(0, len(mcq_styles)-1)], output_path)


def generate():
    llm = 'gpt-4o-mini'
    input_path = '../data/gemini-1.5-flash__v0.jsonl'

    max_workers = 1

    generate_mcq(llm, input_path, max_workers=max_workers, multi_thread=True)

if __name__ == "__main__":
    generate()