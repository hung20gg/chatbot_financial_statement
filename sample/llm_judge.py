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
        
        The answer must accurately reflect the ground truth and the task.
        Return 1 if the answer is correct, 0 otherwise.
        Return in JSON format only.
            
            ```json
            {{
                "correct": 1
            }}
            ```
"""}]
    
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
    history, error_messages, execution_tables = func(**kwargs)
    get_tables = execution_tables[-3:]
    table_text = ""
    for i,table in enumerate(get_tables):
        table_text += f"Table {i+1}\n"
        table_text += utils.df_to_markdown(table)
        table_text += "\n\n"
        
    qa['answer'] = table_text
    qa['code'] = get_code_from_text_response(history[-1]['content'])
    return qa

def scoring_a_task(judge_llm, llm, qa, db, function, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    task = qa['question']
    ground_truth = qa['answer']
    answer = get_answer(function, qa, llm=llm, task=task, db=db, **kwargs)
    
    return llm_judge(judge_llm, task, answer, ground_truth, db, verbose=False)

def scoring(judge_llm, llm, qas, db, function, **kwargs):
    """
    Score a list of tasks based on the qas and the function.
    """
    scores = []
    input_tokens = []
    output_tokens = []
    successful_qa = []
    for qa in qas:
        try:
            
            scores.append(scoring_a_task(judge_llm, llm, qa, db, function, **kwargs))
            usage = llm.usage()
            input_tokens.append(usage['input_token'])
            output_tokens.append(usage['output_token'])
            successful_qa.extend(qa.keys())
            llm.reset_token()
            
        except Exception as e:
            print(e)
    
    df = pd.DataFrame({'score': scores, 'input_token': input_tokens, 'output_token': output_tokens, 'question': successful_qa})
    return df
    
def get_llm(llm_obj, model_name):
    if model_name is not None:
        llm = llm_obj(model_name)
    else:
        llm = llm_obj()
    return llm
    
def scoring_a_task_parallel(judge_llm_obj, llm_obj, qa, db, function, model_name = None, judge_model_name = None, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    judge_llm = get_llm(judge_llm_obj, judge_model_name)
    llm = get_llm(llm_obj, model_name)    
    score = scoring_a_task(judge_llm, llm, qa, db, function, **kwargs)
    usage = llm.usage()
    qa['score'] = score
    qa['input_token'] = usage['input_token']
    qa['output_token'] = usage['output_token']
    
    return qa


def get_a_answer_parallel(llm_obj, qa, db, func, model_name = None, **kwargs):
    llm = get_llm(llm_obj, model_name)
    answer = get_answer(func, qa, llm=llm, db=db, **kwargs)
    return answer


def get_answers_parallel(llm_obj, tasks, db, func, **kwargs):
    answers = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(get_a_answer_parallel, llm_obj, task, db, func, **kwargs): task for task in tasks}
        
        for future in as_completed(future_to_task):
            answers.append(future.result())
            
    return answers


def scoring_parallel(judge_llm_obj, llm_obj, qas, db, function, **kwargs):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
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
    
    conn = connect_to_db(db_name, user, password, host, port)
        
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
    
    db = DBHUB(conn, bank_vector_store, none_bank_vector_store, sec_vector_store, ratio_vector_store, vector_db_company, vector_db_sql)
    print("DB initialized")
    
    with open('../synthetic/gpt-4o_generated.json', 'r') as f:
        questions = json.load(f)
    
    # Test
    questions = questions[:2]
    
    # Change LLM here
    llm_obj = Gemini #ChatGPT
    model_name = 'gemini-1.5-flash-002' #'gpt-4o'
    
    judge_llm_obj = ChatGPT
    judge_model_name = 'gpt-4o'
    
    result = scoring_parallel(judge_llm_obj, llm_obj, questions, db, reasoning_text2SQL, model_name=model_name, judge_model_name=judge_model_name)
    print(result)
    # table = get_answer(reasoning_text2SQL, llm=llm, task=questions[0], db=db)
    #result = get_answers_parallel(llm_obj, questions, db, reasoning_text2SQL, model_name=model_name)
    
    # with open('../synthetic/gpt-4o_generated.json', 'w') as f:
    #     json.dump(result, f, indent=4)