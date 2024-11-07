from setup_db import DBHUB, create_chroma_db, connect_to_db
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_test import reasoning_text2SQL

from llm.llm_utils import get_json_from_text_response, get_code_from_text_response
from llm.llm.chatgpt import ChatGPT, OpenAIWrapper
from llm.llm.gemini import Gemini
import pandas as pd
import utils

import sys
import os
import json

def llm_judge(llm, task, answer, ground_truth, db, verbose=False):
    """
    Judge the llm model and the ground truth. 
    """
    messages = [
        {
            "role": "user",
            "content": f"""
        Your task is to judge the answer of the model with the ground truth.
        
        <task>
        {task}
        </task>
        
        <answer>
        {answer}
        </answer>
        
        <ground_truth>
        {ground_truth}
        </ground_truth>
        
        The response must align accurately with the ground truth and the objectives of the task.
        Analyze carefully the task, question and ground truth.
"""}]
    
    response = llm(messages)
    
    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
    
    messages.append({
        "role": "user",
        "content": """
        Return 1 if the answer is correct, 0.5 for partial accurate, 0 otherwise.
        Return in JSON format.
            
            ```json
            {{
                "correct": 1
            }}
        """
    })
    
    response = llm(messages)
    
    try:
        score = get_json_from_text_response(response, new_method=True)['correct']
    except Exception as e:
        print(e)
        return 0
    
    return score
    
    
# expect the question and ground truth template should be in the following format:
# {
#     "question": "Get the Total Assets of company1",
#     "ground_truth": "The Total Assets of company1 is 1000",
# }

def get_answer(func, qa, **kwargs):
    task = qa['question']
    history, error_messages, execution_tables = func(task = task, **kwargs)
    get_tables = execution_tables[-3:]
    table_text = ""
    for i,table in enumerate(get_tables):
        table_text += f"Table {i+1}\n"
        table_text += utils.df_to_markdown(table)
        table_text += "\n\n"
        
    qa['answer'] = table_text
    qa['code'] = get_code_from_text_response(history[-1]['content'])
    return qa

def get_prediction_answer(func, qa, **kwargs):
    task = qa['question']
    history, error_messages, execution_tables = func(task = task, **kwargs)
    get_tables = execution_tables[-3:]
    table_text = ""
    for i,table in enumerate(get_tables):
        table_text += f"Table {i+1}\n"
        table_text += utils.df_to_markdown(table)
        table_text += "\n\n"
        
    qa['response'] = table_text
    qa['code_response'] = get_code_from_text_response(history[-1]['content'])
    return qa

def scoring_a_task(judge_llm, llm, qa, db, function, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    task = qa['question']
    ground_truth = qa['answer']
    qa = get_prediction_answer(function, qa, llm=llm, db=db, **kwargs)
    answer = qa['response']
    
    qa['evaluate'] = llm_judge(judge_llm, task, answer, ground_truth, db, verbose=False)
    return qa


def get_llm(llm_obj, model_name):
    if model_name is not None:
        llm = llm_obj(model_name)
    else:
        llm = llm_obj()
    return llm

import time    

def scoring_a_task_parallel(judge_llm_obj, llm_obj, qa, db, function, model_name = None, judge_model_name = None, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    judge_llm = get_llm(judge_llm_obj, judge_model_name)
    llm = get_llm(llm_obj, model_name)    
    qa = scoring_a_task(judge_llm, llm, qa, db, function, **kwargs)
    usage = llm.usage()
    
    qa['input_token'] = usage['input_token']
    qa['output_token'] = usage['output_token']
    
    time.sleep(10)
    return qa


def get_a_answer_parallel(llm_obj, qa, db, func, model_name = None, **kwargs):
    llm = get_llm(llm_obj, model_name)
    answer = get_answer(func, qa, llm=llm, db=db, **kwargs)
    time.sleep(10)
    return answer


def get_answers_parallel(llm_obj, tasks, db, func, **kwargs):
    answers = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(get_a_answer_parallel, llm_obj, task, db, func, **kwargs): task for task in tasks}
        
        for future in as_completed(future_to_task):
            answers.append(future.result())
            
    return answers


def scoring_parallel(judge_llm_obj, llm_obj, qas, db, function, **kwargs):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_qa = {executor.submit(scoring_a_task_parallel, judge_llm_obj, llm_obj, qa, db, function, **kwargs): qa for qa in qas}
        
        for future in as_completed(future_to_qa):
            results.append(future.result())
            
    return results


if __name__ == '__main__':
    
    db_name = 'test_db'
    user = 'postgres'
    password = '12345678'
    port = '5433'
    host = 'localhost'
    
        
    collection_chromadb = 'category_bank_chroma'
    persist_directory = 'data/category_bank_chroma'
    bank_vector_store = create_chroma_db(collection_chromadb, persist_directory)

    collection_chromadb = 'category_non_bank_chroma'
    persist_directory = 'data/category_non_bank_chroma'
    none_bank_vector_store = create_chroma_db(collection_chromadb, persist_directory)

    collection_chromadb = 'category_sec_chroma'
    persist_directory = 'data/category_sec_chroma'
    sec_vector_store = create_chroma_db(collection_chromadb, persist_directory)

    collection_chromadb = 'category_ratio_chroma'
    persist_directory = 'data/category_ratio_chroma'
    ratio_vector_store = create_chroma_db(collection_chromadb, persist_directory)

    collection_chromadb = 'company_name_chroma'
    persist_directory = 'data/company_name_chroma'
    vector_db_company = create_chroma_db(collection_chromadb, persist_directory)

    collection_chromadb = 'sql_query'
    persist_directory = 'data/sql_query'
    vector_db_sql = create_chroma_db(collection_chromadb, persist_directory)
    
    conn = {
        'db_name': db_name,
        'user': user,
        'password': password,
        'host': host,
        'port': port
        
    }
    
    db = DBHUB(conn, bank_vector_store, none_bank_vector_store, sec_vector_store, ratio_vector_store, vector_db_company, vector_db_sql)
    
    print("DB initialized")
    
    with open('../synthetic/gpt-4o-generated-v2-scored-pass.json', 'r') as f:
        questions = json.load(f)
    
    # with open('../synthetic/gpt-4o-generated-v2.json', 'r') as f:
    #         old_result = json.load(f)
    # # Test
    questions = questions[:3]
    
    # Change LLM here
    llm_obj = ChatGPT #ChatGPT
    model_name = 'gpt-4o-mini' #'gpt-4o'
    
    judge_llm_obj = Gemini
    judge_model_name = 'gemini-1.5-pro-002'
    
    # result = scoring_parallel(judge_llm_obj, llm_obj, questions, db, reasoning_text2SQL, model_name=model_name, judge_model_name=judge_model_name)
    # print(result)
    # table = get_answer(reasoning_text2SQL, llm=llm, task=questions[0], db=db)
    save_name = 'gpt-4o-mini'
    batch_size = 6
    for i in range(0, len(questions), batch_size):
        bs_questions = questions[i:i+batch_size]
        
        try:
            result = scoring_parallel(judge_llm_obj, llm_obj, questions, db, reasoning_text2SQL, model_name=model_name, judge_model_name=judge_model_name)
            
            if i == 0:
                with open(f'../synthetic/gpt-4o-generated-v2-{save_name}.json', 'w') as f:
                    json.dump(result, f, indent=4)
            else:
                with open(f'../synthetic/gpt-4o-generated-v2-{save_name}.json', 'r') as f:
                    old_result = json.load(f)
                
                old_result.extend(result)
                with open(f'../synthetic/gpt-4o-generated-v2-{save_name}.json', 'w') as f:
                    json.dump(old_result, f, indent=4)
            
        except Exception as e:
            print(e)