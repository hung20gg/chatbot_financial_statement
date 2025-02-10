from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 

from utils import get_available_path, append_jsonl_to_file, get_avaliable_questions
sys.path.append('..')
import pandas as pd
import numpy as np
import json
import time
import os

from agent.text2sql_utils import get_llm_wrapper
from llm.llm_utils import get_json_from_text_response

current_dir = os.path.dirname(os.path.abspath(__file__))


def single_validate_qa(llm, output_path, qa):
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
You are receiving a task and a table of report from your colleague. Your task is to valuate does the table match with the task or not.
Sometime, the report may contain some errors, null or not related to the task.

Ignore the forecasting part of the question, you have to evaluate the report and return 1 only if you can confidently answer the task based on the provided data in the table only with no additional calculation step, 0 otherwise. 

Note:
- Only accept annual data if the quarter is set to 0.

<question>
{task}
</question>

<table>
{table[:5000]}
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

    append_jsonl_to_file(result, output_path)
    
    return qa
    
def parallel_validate_qa(llm, *args):
    llm = get_llm_wrapper(model_name=llm)
    return single_validate_qa(llm, *args)

def evaluate_qa_quality(path, llm_name, multi_thread=False, max_workers=4):
    

        
    basename = path.split('/')[-1]
    output_file_name = llm_name.replace('/', '__') + '-scored-' + basename     
    output_path = os.path.join('../data', output_file_name)  
    
    data = get_avaliable_questions(path, output_path)

    print(f"Number of questions: {len(data)}")

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


def single_evaluate_difficulty(llm, output_path, template, qa):
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

    append_jsonl_to_file(result, output_path)
    


def parallel_evaluate_difficult(llm, *args):
    llm = get_llm_wrapper(model_name=llm)
    return single_evaluate_difficulty(llm, *args)

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


def create_mcq_text(mcq):

    choices = ["A", "B", "C", "D", "E"]
    
    question = mcq['question']

    text_choices = ""

    for i, choice in enumerate(mcq['choices']):
        text_choices += f"{choices[i]}. {choice}\n"
    
    text_choices += f"{choices[-1]}. I don't know"

    answer = choices[mcq['answer']]

    return {
        'mcq_question': question,
        'choice': text_choices,
        'answer': answer,
    }

def merge_mcq_and_sql(mcq, sql):
    
    mcq_dict = {}
    for ques in mcq:
        mcq_dict[ques['ids']] = ques
    
    print("Number of SQL questions: ", len(sql))

    sql_dict = {}
    for ques in sql:
        sql_dict[ques['ids']] = ques

    print("Number of MCQ questions: ", len(mcq))

    questions = []
    for sql_question in sql:
        ids = sql_question['ids']
        if ids in mcq_dict.keys():
            question = create_mcq_text(mcq_dict[ids])
            question['ids'] = ids
            question['table'] = sql_question['table']

            questions.append(question)
    
    # Test
    # questions = questions[:10]

    return questions
            


import re

def single_scoring_mcq(llm, response, output_path):
    llm = get_llm_wrapper(model_name=llm)

    mcq = response['mcq_question']
    choice = response['choice']
    correct_choice = response['answer']
    table = response['table']

    system_prompt = """
You will be given a multiple-choice question with 5 choices, and a reference table in the following format:

<question>
{question}
</question>

<table>
{table}
</table>

<choices>
{choices}
</choices>

Notice that there is only one correct answer. If you cannot derive the answer, return choice E.
You will loss 1 point for each wrong answer and gain 1 point for correct answer, however, you will not be penalized if you choose choice E.
So, you can choose to skip the question if you are not sure about the answer.

Analyze carefully and return your choice (A, B, C, D, E) in {$choice} format. For example: {A}.
"""

    prompt = f"""
<question>
{mcq}
</question>

<table>
{table}
</table>

<choices>
{choice}
</choices>
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

    answer = llm(messages)
    cant_answer = False

    try:
        pattern = r'\{(.*?)\}'
        choice = re.search(pattern, answer).group(1)

        # Clean the choice
        choice = choice.upper().replace(' ', '').replace('$','').replace('\n', '').replace('\t', '').replace('.', '').replace(',', '').replace('(', '').replace(')', '').strip()

        print(f"Choice: {choice}")
        print(f"Correct choice: {correct_choice}")

        if choice == correct_choice:
            score = 1
        elif choice == 'E':
            score = 0
            cant_answer = True
        else:
            score = 0
    except:
        score = 0
        cant_answer = True

    result = dict()
    result['ids'] = response['ids']
    result['score'] = score
    result['cant_answer'] = cant_answer

    append_jsonl_to_file(result, output_path)

    return result


def scoring_mcq(llm, responses, output_path, max_workers=4, multi_thread=False):
    
    total_questions = len(responses)
    results = []

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(single_scoring_mcq, llm, response, output_path): response for response in responses}
            for future in as_completed(future_to_task):
                response = future_to_task[future]
                results.append(response)
    else:
        for response in responses:
            results.append(single_scoring_mcq(llm, response, output_path))
    
    score = 0
    for result in results:
        try:
            score += result['score']
        except:
            pass

    print(f"===== Total questions: {total_questions} =====")
    print(f"===== Out of responsed answer, Score: {score/total_questions} =====")

    return results


def evaluate_sql_generation(sql_path, mcq_path, llm_name, multi_thread=False, max_workers=4):
    
    sql_data = []
    with open(sql_path) as f:
        for line in f:
            sql_data.append(json.loads(line))
    
    mcq_data = []
    with open(mcq_path) as f:
        for line in f:
            mcq_data.append(json.loads(line))
    
    output_file_name = llm_name.replace('/', '__') + '-evaluate-' + os.path.basename(sql_path)
    output_path = os.path.join('../data', output_file_name)

    output_path = get_available_path(output_path)

    responses = merge_mcq_and_sql(mcq_data, sql_data)

    return scoring_mcq(llm_name, responses, output_path, max_workers, multi_thread)


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate QA quality')
    parser.add_argument('--task', type=str, default='qa_quality', help='task to evaluate')
    parser.add_argument('--path', type=str, default=os.path.join(current_dir, '../data/deepseek-chat__v0_good.jsonl'), help='Path to the generated QA')
    parser.add_argument('--mcq_path', type=str, default=os.path.join(current_dir, '../data/mcq_v0.jsonl'), help='Path to the generated MCQ')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='LLM model name')
    parser.add_argument('--multi_thread', type=bool, default=False, help='Use multi-threading or not')
    parser.add_argument('--template', type=str, default='vertical', help='Template for the difficulty evaluation')
    parser.add_argument('--max_workers', type=int, default=4, help='Max workers for multi-threading')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()

    if args.task == 'qa_quality':
        evaluate_qa_quality(args.path, args.llm, args.multi_thread, args.max_workers)
    elif args.task == 'qa_difficulty':
        evaluate_qa_difficult(args.path, args.llm, args.multi_thread, args.template, args.max_workers)
    elif args.task == 'evaluate':
        evaluate_sql_generation(args.path, args.mcq_path, args.llm, args.multi_thread, args.max_workers) 
    else:
        raise ValueError('Task not found')