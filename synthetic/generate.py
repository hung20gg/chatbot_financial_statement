from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 
sys.path.append('..')
import pandas as pd
import numpy as np

from llm.llm.gemini import Gemini
from llm.llm_utils import get_json_from_text_response

main_tasks = [
    "Financial Ratios", 
    "Accounts in Financial Statements", 
    "Both Financial Ratios and Accounts in Financial Statements", 
    #"DuPont Analysis",
]

sub_tasks = [
    #"get data 1 or more company", 
    "compare 2 or more company", 
    "analyze the Subsidiaries or invested company", 
   # "analyze over industry average report", "analyze company with its industry average report", 
    "Ranking (Top 5 - Top 10)", 
    "Ranking with special criterias (Top 5 - Top 10)(e.g: total asset >100B VND,  ROE > 20%)",
]

analyzing_types = [
    #"General assessment of the financial position",
    "Analysis of the financial structure",
    "Analysis of liquidity and payment status",
    "Analysis of financial risk",
    "Analysis of financial equilibrium",
    "Analysis of business performance",
    "Analysis of profitability",
    "Cash flow analysis",
    "Forecasting of financial indicators",
    #"Business valuation"
]

times = [
    "at specific year", 
    "at specific quater and year", 
    "over time with specific period",
]
job_titles = [
    "Auditor",
    "Financial Analyst",
    "Investment Analyst",
    "Financial Manager",
    "Accountant",
    "Tax Consultant",
    "Financial Consultant",
    "Broker"

]

def generate_questions(llm, main_task, sub_task, analyzing_types, time, job_titles,company_table):
    system_prompt = f"You are a/an {job_titles}, you are having a task of analyzing the information from financial reports of companies from 2020 to Q2 2024 to give the good insights. You have deep knowledge about the domain of financial analyzing. Your questions should contain popular ratios, accounts in financial reports analysis."
   # , and you have to ask many explicit,meaningful and insightful questions about the financial reports of companies.
    
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': f"""Here are the details of the companies that you will be analyzing:
{company_table}
Notice that there are 4 company that was not listed on any exchange, since they are government company, and they only have data about ownership/shareholder of other companies. 

Task: Generate only 2 questions on {main_task} with {sub_task},{time}, and the questions need to be diversifying within {analyzing_types}, the question contents and remember that each time of question generation needs to be diverse in content. The questions should be concise. 

Note:
- You must ask questions to provide data only.
- Your question must only contain the name of the companies. You must not leak any other information of the company table.
- It is recommended not to provide stock code of companies in the generated questions.
- you must return questions only.
- You can ask questions in any format, but the questions must be relevant to the task.
- You mustn't contain the word : "Include" the list of companies in the end of the question.
- your question should not contain prediction or forecast parts.
- your questions should not contain the phrase: "This will help", "consider".
- your questions should not contain industry comparison.

Return the questions in a JSON format
{{
    "questions": ["Question 1", "Question 2"]
}}

"""
        }
    ]
    response = llm(messages, temperature= 0.4)
    
    output = {
        'main_task': main_task,
        'sub_task': sub_task,
        'time': time,
        'analyzing_types': analyzing_types,
        'job_titles': job_titles,
        'questions': []
    }
    
    try:
        output['questions'] = get_json_from_text_response(response, new_method=True)['questions']
    
    except Exception as e:
        print(e)
        
    return output
        
def parallel_generate_questions(*args):
    llm = Gemini()
    return generate_questions(llm, *args)


def main():
    
    df = pd.read_csv('../csv/df_company_info.csv')
    df_profile = df[['stock_code','en_short_name', 'industry', 'exchange']]
    df_sub = pd.read_csv('../csv/df_sub_and_shareholders.csv')
    df_profile['Has Subsidiaries/ Invest on other company'] = df_profile['stock_code'].apply(lambda x: 'Yes' if x in df_sub['stock_code'].values else 'No')
    company_table = df_profile.to_markdown()
    
    tasks = []
    for main_task in main_tasks:
        for sub_task in sub_tasks:
            for analyzing_type in analyzing_types:
                for time in times:
                    for job_title in job_titles:
                        tasks.append((main_task, sub_task,analyzing_type, time,job_title, company_table))
                
    # Test
    tasks = random.sample(tasks,25)
    
    
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_task = {executor.submit(parallel_generate_questions, *args): (args) for args in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Task completed")
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    return results

if __name__ == "__main__":
    generated_questions = main()
    import json
    with open('generated_questions.json', 'w') as f:
        json.dump(generated_questions, f, indent=4)

