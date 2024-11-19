from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append('..')
from llm_test import reasoning_text2SQL
from branch_reasoning import llm_branch_reasoning
import utils


from llm.llm_utils import get_json_from_text_response, get_code_from_text_response
from llm.llm.chatgpt import ChatGPT, OpenAIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from llm.llm.gemini import Gemini
from ETL import setup_db

import os
import json
from dotenv import load_dotenv
load_dotenv()

def llm_judge(llm, task, answer, ground_truth):
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
        {answer[:8000]}
        </answer>
        
        <ground_truth>
        {ground_truth}
        </ground_truth>
  
<task>      
Analyze carefully the task, question and ground truth and score the answer accordingly.
</task>

The response must align accurately with the ground truth and the objectives of the task.

Score the answer based on the following criteria:
 - 1 if the response provide accurate and enough answer to the task.
 - 0.5 for accurate answer, with some abundant information to the task (provide some unnecessary information).
 - 0 otherwise.
  
Return in JSON format.
            
            ```json
            {{
                "correct": 1
            }}      
        
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
    try:
        history, error_messages, execution_tables = func(task = task, **kwargs)
        get_tables = execution_tables[-3:]
        table_text = ""
    except Exception as e:
        print(e)
        qa['answer'] = "Error in getting the answer"
        qa['code'] = "Error in getting the code"
        return qa
    
    for i,table in enumerate(get_tables):
        table_text += f"Table {i+1}\n"
        table_text += utils.df_to_markdown(table)
        table_text += "\n\n"
        
    qa['answer'] = table_text
    qa['code'] = get_code_from_text_response(history[-1]['content'])
    return qa

def get_prediction_answer(func, qa, **kwargs):
    task = qa['question']
    start = time.time()
    
    try:
        history, error_messages, execution_tables = func(task = task, **kwargs)
        get_tables = execution_tables[-3:]
        
    except Exception as e:
        print(e)
        qa['response'] = "Error in getting the answer"
        qa['code_response'] = "Error in getting the code"
        return qa
    
    table_text = ""
    end = time.time()
    
    for i,table in enumerate(get_tables):
        table_text += f"Table {i+1}\n"
        table_text += utils.df_to_markdown(table)
        table_text += "\n\n"
        
    qa['response'] = table_text[:8000]
    qa['code_response'] = get_code_from_text_response(history[-1]['content'])
    qa['time'] = end - start
    return qa

def scoring_a_task(judge_llm, llm, qa, db, function, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    task = qa['question']
    ground_truth = qa['answer']
    qa = get_prediction_answer(function, qa, llm=llm, db=db, **kwargs)
    answer = qa['response']
    
    qa['evaluate'] = llm_judge(judge_llm, task, answer, ground_truth)
    return qa


def get_llm(llm_obj, model_name, **kwargs):
    if model_name is not None:
        llm = llm_obj(model_name = model_name, **kwargs)
    else:
        llm = llm_obj(**kwargs)
    return llm

import time    

def scoring_a_task_parallel(judge_llm, llm, qa, db, function, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    # judge_llm = get_llm(judge_llm_obj, judge_model_name, **kwargs)
    # llm = get_llm(llm_obj, model_name, **kwargs)    
    qa = scoring_a_task(judge_llm, llm, qa, db, function, **kwargs)
    usage = llm.usage()
    
    qa['input_token'] = usage['input_token']
    qa['output_token'] = usage['output_token']
    llm.reset_token()
    
    time.sleep(2)
    return qa


def get_a_answer_parallel(llm, qa, db, func, **kwargs):
    # llm = get_llm(llm_obj, model_name, **kwargs)
    answer = get_answer(func, qa, llm=llm, db=db, **kwargs)
    time.sleep(2)
    return answer


def get_answers_parallel(llm, tasks, db, func, **kwargs):
    answers = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(get_a_answer_parallel, llm, task, db, func, **kwargs): task for task in tasks}
        
        for future in as_completed(future_to_task):
            answers.append(future.result())
            
    return answers


def scoring_parallel(judge_llm, llm, qas, db, function, **kwargs):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_qa = {executor.submit(scoring_a_task_parallel, judge_llm, llm, qa, db, function, **kwargs): qa for qa in qas}
        
        for future in as_completed(future_to_qa):
            results.append(future.result())
            
    return results

def scoring(judge_llm, llm, qas, db, function, **kwargs):
    results = []
    for qa in qas:
        result = scoring_a_task_parallel(judge_llm, llm, qa, db, function, **kwargs)
        results.append(result)
        
    return results

if __name__ == '__main__':
    
    model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': 'cuda'})
    
    db = setup_db(model)
    
    print("DB initialized")
    
    with open('../synthetic/gpt-4o-generated-v2-scored.json', 'r') as f:
        questions = json.load(f)
    
    
    questions = questions
    
    # Change LLM here
    # llm = Gemini() #ChatGPT, OpenAIWrapper
    model_name = 'nvidia/Llama-3.1-Nemotron-70B-Instruct' #'gpt-4o'
    api_key = os.getenv('DEEPINFRA_TOKEN')
    host = 'https://api.deepinfra.com/v1/openai'
    llm = OpenAIWrapper(model_name = model_name, api_key = api_key, host = host)
    
    judge_llm = Gemini('gemini-1.5-pro-002')
    
    print(os.getenv('GENAI_API_KEY'))
    
    save_name = 'nemetron-simple-v2'
    batch_size = 12
    results = []
    loop = 2
    
    for _ in range(loop):
        flag = False
        for i in range(0, len(questions), batch_size):
            bs_questions = questions[i:i+batch_size]
            
            try:
                result = scoring_parallel(judge_llm, llm, bs_questions, db, reasoning_text2SQL)    
                results.extend(result)
                
                with open(f'../synthetic/gpt-4o-v2-{save_name}.json', 'w') as f:
                    json.dump(results, f, indent=4)

                
            except Exception as e:
                print(e)
                flag = True
                break
        if flag:
            print("Error in the loop")
            break