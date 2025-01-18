from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 
sys.path.append('..')
import pandas as pd
import numpy as np
import json
import time
import os

from agent.text2sql_utils import get_llm_wrapper
from llm.llm_utils import get_json_from_text_response

current_dir = os.path.dirname(os.path.abspath(__file__))

def append_json_to_file(json_obj, file_path):
    with open(file_path, 'a') as f:
        json.dump(json_obj, f)
        f.write('\n')


def validate_qa(llm, output_path, qa):
    task = qa['question']
    table = qa['table']
    
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

Ignore the forecasting part of the question, you have to evaluate the report and return 1 if you can confidently answer the task with the provided data in the table, 0 otherwise. 

<question>
{task}
</question>

<table>
{table}
</table>

Return final answer in JSON format.
                
    ```json
    {{
        "score": 1
    }}
    ```

"""
        }   
    ]
    
    response = llm(messages)
    try:
        score = get_json_from_text_response(response, new_method=True)['score']
        qa['score'] = score
    except Exception as e:
        print(e)
        qa['score'] = 0

    result = dict()
    result['ids'] = qa['ids']
    result['score'] = qa['score']

    append_json_to_file(result, output_path)
    
    return qa
    
def parallel_validate_qa(llm, *args):
    llm = get_llm_wrapper(model_name=llm)
    return validate_qa(llm, *args)

def evaluate_qa_quality(path, llm_name, multi_thread=False, max_workers=4):
    
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
            
    path = path.split('/')[-1]
    output_file_name = llm_name.replace('/', '__') + '-scored-' + path     
    output_path = os.path.join('../data', output_file_name)  
    
    results = []

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(parallel_validate_qa, llm_name, output_path, qa): qa for qa in data}
            
            for future in as_completed(future_to_task):
                qa = future_to_task[future]
                results.append(qa)

    else:
        for qa in data:
            results.append(parallel_validate_qa(llm_name, output_path, qa))        

    return results


def _evaluate_difficulty(llm, output_path, template, qa):
    task = qa['question']
    code = ""

    for code_block in qa['sql']:
        code += f"```\n\n{code_block}\n\n```" + '\n\n'
    
    system_prompt = f"""
You will be given a question and an SQL query. Your task is to evaluate the difficulty of the question and the SQL query, follow the difficulty of Spider 1.0 dataset. The difficulty should be in the range of 1 to 4, where 1 is the easiest and 4 is the hardest.

Here are the descriptions of tables in the database:

### PostgreSQL tables, with their properties

```sql
{template}
```

<example>
Here is the example of the difficulty evaluation from Spider 1.0 dataset:

### Easy (1)
What is the number of cars with more than 4 cylinders?

```sql
SELECT COUNT(*)
FROM cars_data
WHERE cylinders > 4;
```

### Medium (2)
For each stadium, how many concerts are there?

```sql
SELECT T2.name, COUNT(*)
FROM concert AS T1 JOIN stadium AS T2
ON T1.stadium_id = T2.stadium_id
GROUP BY T1.stadium_id;
```

### Hard (3)
Which countries in Europe have at least 3 car manufacturers?

```sql
SELECT T1.country_name
FROM countries AS T1 JOIN continents AS T2
ON T1.continent = T2.cont_id
JOIN car_makers AS T3
ON T1.country_id = T3.country
WHERE T2.continent = 'Europe'
GROUP BY T1.country_name
HAVING COUNT(*) >= 3;
```

### Extra Hard (4)
What is the average life expectancy in the countries where English is not the official language?

```sql
SELECT AVG(life_expectancy)
FROM country
WHERE name NOT IN
(SELECT T1.name
 FROM country AS T1 JOIN country_language AS T2
 ON T1.code = T2.country_code
 WHERE T2.language = 'English'
 AND T2.is_official = 'T');
```

</example>

"""
    prompt = f"""
<task>
You are given the following question
<question>
{task}
</question>

Here is the SQL query to solve the problem:
<answer>
```sql
    {code}
```
</answer>

Based on the evaluation of Spider 1.0 dataset, evaluate the difficulty of the question and the SQL query. The difficulty should be in the range of 1 to 4, where 1 is the easiest and 4 is the hardest.

Think step-by-step and Return the difficulty in JSON format.

```json
{{
    "difficulty": 1
}}
```

</task>

"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    if qa.get('difficulty', -1) > 0:
        return qa
    
    response = llm(messages)
    try:
        difficulty = get_json_from_text_response(response, new_method=True)['difficulty']
        qa['difficulty'] = difficulty
    except Exception as e:
        print(e)
        qa['difficulty'] = -1

    result = dict()
    result['ids'] = qa['ids']
    result['difficulty'] = qa['difficulty']

    append_json_to_file(result, output_path)
    


def parallel_evaluate_difficult(llm, *args):
    llm = get_llm_wrapper(model_name=llm)
    return _evaluate_difficulty(llm, *args)

def evaluate_qa_difficult(path, llm_name, multi_thread=False, template='vertical', max_workers=4):
        
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    
    results = []

    if template == 'vertical':
        with open('vertical.sql', 'r') as f:
            template = f.read()

    elif template == 'horizontal':
        with open('horizontal.sql', 'r') as f:
            template = f.read()
    else:
        raise ValueError('Template must be either vertical or horizontal')

    output_file_name = llm_name.replace('/', '__') + '-difficulty-' + path.split('/')[-1]
    output_path = os.path.join('../data', output_file_name)

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(parallel_evaluate_difficult, llm_name, output_path, template, qa): qa for qa in data}
            for future in as_completed(future_to_task):
                qa = future_to_task[future]
                results.append(qa)
    else:
        for qa in data:
            results.append(parallel_evaluate_difficult(llm_name, output_path, template, qa))

    return results


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate QA quality')
    parser.add_argument('--task', type=str, default='qa_quality', help='task to evaluate')
    parser.add_argument('--path', type=str, default=os.path.join(current_dir, '../data/gpt-4o-mini__v0.jsonl'), help='Path to the generated QA')
    parser.add_argument('--llm', type=str, default='gpt-4o', help='LLM model name')
    parser.add_argument('--multi_thread', type=bool, default=False, help='Use multi-threading or not')
    parser.add_argument('--template', type=str, default='vertical', help='Template for the difficulty evaluation')
    parser.add_argument('--max_workers', type=int, default=4, help='Output file')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()

    if args.task == 'qa_quality':
        evaluate_qa_quality(args.path, args.llm, args.multi_thread, args.max_workers)
    elif args.task == 'qa_difficulty':
        evaluate_qa_difficult(args.path, args.llm, args.multi_thread, args.template, args.max_workers)
