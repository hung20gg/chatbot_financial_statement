from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 
sys.path.append('..')
import pandas as pd
import numpy as np
import json
import time

from llm.llm.gemini import Gemini
from llm.llm_utils import get_json_from_text_response


def validate_qa(llm, qa):
    task = qa['question']
    answer = qa['answer']
    
    system_prompt = f"""You are an auditor, and you have to evaluate the report from your colleague."""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
You are receiving a task and a table of report from your colleague. Your task is to evaluate does the table related to the task or not.
Sometime, the report may contain some errors, null or not related to the task.

You have to evaluate the report and return 1 if the table is related to the task, 0 otherwise. 

<question>
{task}
</question>

<answer>
{answer}
</answer>

Return final answer in JSON format.
                
    ```json
    {{
        "correct": 1
    }}
    ```

"""
        }   
    ]
    
    response = llm(messages)
    try:
        score = get_json_from_text_response(response, new_method=True)['correct']
        qa['score'] = score
    except Exception as e:
        print(e)
        qa['score'] = 0
        
    return qa
    
def parallel_validate_qa(*args):
    llm = Gemini('gemini-1.5-pro-002')
    return validate_qa(llm, *args)

def evaluate_qa_quality():
    
    with open('gpt-4o-generated-v2.json') as f:
        data = json.load(f)
    
    
    
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(parallel_validate_qa, qa): qa for qa in data}
        
        for future in as_completed(future_to_task):
            qa = future_to_task[future]
            results.append(qa)
            
    return results

if __name__ == '__main__':
    results = evaluate_qa_quality()
    with open('gpt-4o-generated-v2-scored.json', 'w') as f:
        json.dump(results, f, indent=4)