from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 
sys.path.append('..')
import pandas as pd
import numpy as np
import random
import json

from llm.llm.gemini import Gemini
from llm.llm_utils import get_json_from_text_response

import os 
from dotenv import load_dotenv
load_dotenv()


df = pd.read_csv('../csv/df_company_info.csv')
df_profile = df[['stock_code','en_short_name', 'industry', 'exchange', 'stock_indices']]
df_sub = pd.read_csv('../csv/df_sub_and_shareholders.csv')
df_profile['Has Subsidiaries/ Invest on other company'] = df_profile['stock_code'].apply(lambda x: 'Yes' if x in df_sub['stock_code'].values else 'No')
df.drop(columns=['stock_code'], inplace=True)
company_table = df_profile.to_markdown()


main_tasks = [
    "Financial Ratios", 
    "Accounts in Financial Statements (including Explaination parts)", 
    "Both Financial Ratios and Accounts in Financial Statements", 
    # "DuPont Analysis",
]

sub_tasks = [
    "get data 1 or more company", 
    "compare 2 or more company", 
    "analyze the Subsidiaries or invested company", 
    "analyze over industry average report (might be % of X in the industry)", 
    "analyze company with its industry average ratio", 
    "compare within the same exchange or stock indices",
    "Ranking top 1-5-10 with or without special criterias (Top 5 - Top 10)(e.g: total asset >100B VND,  ROE > 20%)",
]

analyzing_types = [
    # "General assessment of the financial position",
    "Analysis of the financial structure",
    "Analysis of liquidity and payment status",
    "Analysis of financial risk",
    "Analysis of financial equilibrium",
    "Analysis of business performance",
    "Analysis of profitability",
    "Cash flow analysis",
    "Analysis of capital structures",
    "Analysis of loan types (mostly for banks, focus on loan to customer, loan type, duration, etc)",
    "Analysis of financial explaination details (bank loan, bond, etc)",
    # "Forecasting of financial indicators",
    # "Business valuation"
]

times = [
    "at specific year, optionally include quater", 
    "over time with specific period",
]
job_titles = [
    "Auditor",
    "Financial Analyst",
    # "Investment Analyst",
    # "Financial Manager",
    # "Accountant",
    "Financial Consultant",
    # "Broker"
]

file_path = 'generated_questions.json'
# Function to add new content to a JSON file
def add_content_to_json(new_data):
    try:
        # Read existing data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Append or merge new data
        if isinstance(data, list):
            data.append(new_data)  # Append if JSON data is a list
        elif isinstance(data, dict):
            data.update(new_data)  # Merge if JSON data is a dictionary

        # Write updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        print("New content added successfully.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: Invalid JSON file or file not found.")


def generate_questions(llm, main_task, sub_task, analyzing_types, time, job_titles):
    system_prompt = f"""You are a/an {job_titles}, you are having a task of analyzing the information from financial reports of companies from 2020 to Q3 2024 to give the good insights. You have deep knowledge about the domain of financial analyzing. Your questions should contain financial ratios, accounts in financial reports analysis.

     
Note:
- You must ask questions to provide data only.
- Your question must contain the name of the companies. You must not leak any other information of the company table beside company name.
- you must return questions only.
- You can ask questions in any format, but the questions must be relevant to the task.
- You mustn't contain the word : "Include" the list of companies in the end of the question.
- your question should not contain prediction or forecast parts.
- your questions should not contain the phrase: "This will help", "consider".
- Making the question from easy to hard and more complex for each request. (Q1 is easy and final question is the most difficult)
"""
   # , and you have to ask many explicit,meaningful and insightful questions about the financial reports of companies.
    
    if isinstance(analyzing_types, str):
        analyzing_types = [analyzing_types]
    if isinstance(job_titles, str):
        job_titles = [job_titles]
    if isinstance(sub_task, str):
        sub_task = [sub_task]
    if isinstance(time, str):
        time = [time]
    if isinstance(main_task, str):
        main_task = [main_task]
    
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

Task: Generate 3-5 questions on {main_task[0]} with {sub_task[0]},{time[0]}, and the questions need to be diversifying within {analyzing_types[0]}, the question contents and remember that each time of question generation needs to be diverse in content. The questions should be concise. 

Return the questions in a JSON format

```json
    {{
        "questions": ["Question 1", "Question 2"]
    }}
```

"""
        }
    ]
    response = llm(messages, temperature= 0.6 + random.random() * 0.3)
    
    questions = []
    output = {
        'main_task': main_task[0],
        'sub_task': sub_task[0],
        'time': time[0],
        'analyzing_types': analyzing_types[0],
        'job_titles': job_titles[0],
        'questions': []
    }
    
    try:
        output['questions'] = get_json_from_text_response(response, new_method=True)['questions']
    
    except Exception as e:
        print(e)
        
    questions.append(output)

    messages.append({
        'role': 'assistant',
        'content': response
    })

    for i in range(len(main_task)-1):
        messages.append(
            {
                'role': 'user',
                'content': f"""Task: Generate only 2 questions on {main_task[i+1]} with {sub_task[i+1]},{time[i+1]}, and the questions need to be diversifying within {analyzing_types[i+1]}, the question contents and remember that each time of question generation needs to be diverse in content. The questions should be concise."""
            }
        )

        response = llm(messages, temperature= 0.6 + random.random() * 0.3)
        output = {
            'main_task': main_task[i+1],
            'sub_task': sub_task[i+1],
            'time': time[i+1],
            'analyzing_types': analyzing_types[i+1],
            'job_titles': job_titles[i+1],
            'questions': []
        }

        try:
            output['questions'] = get_json_from_text_response(response, new_method=True)['questions']
        except Exception as e:
            print(e)
        questions.append(output)

        messages.append({
            'role': 'assistant',
            'content': response
        })
    
    add_content_to_json(questions)
    return questions



        
def parallel_generate_questions(*args):
    llm = Gemini(model_name="gemini-1.5-flash")

    return generate_questions(llm, *args)


def main():
    
    
    
    tasks = []
    for main_task in main_tasks:
        for sub_task in sub_tasks:
            for analyzing_type in analyzing_types:
                for time in times:
                    for job_title in job_titles:
                        tasks.append((main_task, sub_task,analyzing_type, time,job_title))

    BATCH_SIZE = 10
    # batch_tasks = [tasks[i:i+BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]

    batch_tasks = []

    for i in range(0, len(tasks), BATCH_SIZE):
        batch_task = [[], [], [], [], []]
        for task in tasks[i:i+BATCH_SIZE]:
            for j in range(5):
                
                batch_task[j].append(task[j])
        batch_tasks.append(batch_task)

    # Test

    batch_tasks = random.sample(batch_tasks, 80)
    # batch_tasks = batch_tasks[:2]

    print(f"Number of tasks: {len(tasks)}")
     
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(parallel_generate_questions, *args): (args) for args in batch_tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.extend(result)
                print(f"Task completed")
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    return results

if __name__ == "__main__":
    with open(file_path, 'w') as f:
        json.dump([], f, indent=4)
    generated_questions = main()
    
    with open('temp_generated_questions.json', 'w') as f:
        json.dump([generated_questions], f, indent=4)

    # print(os.getenv('GENAI_API_KEY'))

