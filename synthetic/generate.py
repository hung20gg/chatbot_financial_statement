from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append('..')
import pandas as pd
import numpy as np

from llm.llm.gemini import Gemini
from llm.llm_utils import get_json_from_text_response

main_tasks = [
    "Financial Ratios", "Accounts in Financial Statements", "Both Financial Ratios and Accounts in Financial Statements", "DuPont Analysis",
]

sub_tasks = [
    "get data 1 or more company", "compare 2 or more company", "analyze the Subsidiaries or invested company", "analyze over industry average report", "analyze company with its industry average report", "Ranking (Top 5 - Top 10)", "Ranking with special criterias (Top 5 - Top 10)(e.g: total asset >100B VND,  ROE > 20%)",
]

times = [
    "at specific year", "at specific quater and year", "over time with specific period",
]

def generate_questions(llm, main_task, sub_task, time, company_table):
    system_prompt = "You are a head of the investment fund and you are trying give tasks to your team to analyze the information from financial report of company from 2020 to Q2 2024."
    
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': f"""Here are the details of the company
{company_table}
Notice that there are 4 company that was not listed on any exchange, since they are government company, and they only have data about ownership/shareholder of other companies. 

Task: Generate 2 questions on {main_task} with {sub_task} {time}. 
Note:
- You must ask questions to provide data only.
- Your question must only contain the name of the company and/or the industry only. You must only leak any other information of the company table.
- It is recommended not to provide stock code.

Return the questions in a JSON format
{{
    "questions": []
}}

"""
        }
    ]
    response = llm(messages)
    
    output = {
        'main_task': main_task,
        'sub_task': sub_task,
        'time': time,
        'questions': []
    }
    
    try:
        output['questions'] = get_json_from_text_response(response, new_method=True)['questions']
    
    except Exception as e:
        print(e)
        
    return output
        
def parallel_generate_questions(main_task, sub_task, time, company_table):
    llm = Gemini()
    return generate_questions(llm, main_task, sub_task, time, company_table)


def main():
    
    df = pd.read_csv('../csv/df_company_info.csv')
    df_profile = df[['stock_code','en_short_name', 'en_company_name', 'industry', 'exchange']]
    df_sub = pd.read_csv('../csv/df_sub_and_shareholders.csv')
    df_profile['Has Subsidiaries/ Invest on other company'] = df_profile['stock_code'].apply(lambda x: 'Yes' if x in df_sub['stock_code'].values else 'No')
    company_table = df_profile.to_markdown()
    
    tasks = []
    for main_task in main_tasks:
        for sub_task in sub_tasks:
            for time in times:
                tasks.append((main_task, sub_task, time))
                
    # Test
    tasks = tasks[:5]
    
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(parallel_generate_questions, main_task, sub_task, time, company_table): (main_task, sub_task, time) for main_task, sub_task, time in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Task {task} completed")
            except Exception as exc:
                print(f"Task {task} generated an exception: {exc}")

    return results

if __name__ == "__main__":
    generated_questions = main()
    import json
    with open('generated_questions.json', 'w') as f:
        json.dump(generated_questions, f, indent=4)

