from setup_db import DBHUB
from concurrent.futures import ThreadPoolExecutor, as_completed


from llm.llm_utils import get_json_from_text_response
from llm.llm.chatgpt import ChatGPT, OpenAIWrapper
from llm.llm.gemini import Gemini
import pandas as pd

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


def scoring_a_task(llm, qa, db, function, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    task = qa['question']
    ground_truth = qa['ground_truth']
    answer = function(llm=llm, task=task, db=db, **kwargs)
    
    return llm_judge(llm, task, answer, ground_truth, db, verbose=False)

def scoring(llm, qas, db, function, **kwargs):
    """
    Score a list of tasks based on the qas and the function.
    """
    scores = []
    input_tokens = []
    output_tokens = []
    successful_qa = []
    for qa in qas:
        try:
            
            scores.append(scoring_a_task(llm, qa, db, function, **kwargs))
            usage = llm.usage()
            input_tokens.append(usage['input_token'])
            output_tokens.append(usage['output_token'])
            successful_qa.extend(qa.keys())
            llm.reset_token()
            
        except Exception as e:
            print(e)
    
    df = pd.DataFrame({'score': scores, 'input_token': input_tokens, 'output_token': output_tokens, 'question': successful_qa})
    return df
    
    
def scoring_a_task_parallel(llm_obj, qa, db, function, **kwargs):
    """
    Score a task based on the qa and the function.
    """
    llm = llm_obj()
    score = scoring_a_task(llm, qa, db, function, **kwargs)
    usage = llm.usage()
    return score, usage['input_token'], usage['output_token'], qa['question']

def scoring_parallel(llm_obj, qas, db, function, **kwargs):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_qa = {executor.submit(scoring_a_task_parallel, llm_obj, qa, db, function, **kwargs): qa for qa in qas}
        
        for future in as_completed(future_to_qa):
            results.append(future.result())
            
    df = pd.DataFrame(results, columns=['score', 'input_token', 'output_token', 'question'])
    return df


if __name__ == '__main__':
    pass