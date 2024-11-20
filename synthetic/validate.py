from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random 
sys.path.append('..')
import pandas as pd
import numpy as np
import json
import time

from llm.llm.gemini import Gemini
from llm.llm.chatgpt import ChatGPT
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


def evaluate_difficult(llm, qa):
    task = qa['question']
    code = qa['code'][-1]['code']
    
    prompt = f"""
Here are the descriptions of tables in the database:

### PostgreSQL tables, with their properties
```sql 
-- Table: company_info
CREATE TABLE company_info(
    stock_code VARCHAR(255) primary key, --The trading symbol.
    is_bank BOOLEAN, --Bool checking whether the company is a bank or not.
    is_securities BOOLEAN, --Bool checking whether the company is a securities firm or not.
    industry VARCHAR(255), --Current industry of company. 
    issue_share int --Number of share issued.
);

-- Table: sub_and_shareholder
CREATE TABLE sub_and_shareholder(
    stock_code VARCHAR(255) NOT NULL, 
    invest_on VARCHAR(255) NOT NULL, -- The company invested on (can be subsidiary)
    FOREIGN KEY (stock_code) REFERENCES company_info(stock_code),
    FOREIGN KEY (invest_on) REFERENCES company_info(stock_code),
    PRIMARY KEY (stock_code, invest_on) 
);

-- Table: map_category_code_bank
CREATE TABLE map_category_code_bank(
    category_code VARCHAR(255) primary key, --The category_code recorded in the financial report.
    en_caption VARCHAR(255), --The Caption for the `category_code`.
    report_type VARCHAR(255) --Report type recorded for each line (balance_sheet, cash_flow_statement or income_statement)
);

-- Table: map_category_code_non_bank. Same as `map_category_code_bank`
CREATE TABLE map_category_code_non_bank(
    category_code VARCHAR(255) primary key,
    en_caption VARCHAR(255),
    report_type VARCHAR(255)
);

-- Table: map_category_code_securities. Same as `map_category_code_bank`
CREATE TABLE map_category_code_securities(
    category_code VARCHAR(255) primary key,
    en_caption VARCHAR(255),
    report_type VARCHAR(255)
);

-- Table: bank_financial_report: Financial report of banks
CREATE TABLE bank_financial_report(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int, -- The reported financial year
    quarter int, --  The quarter reported (contain value either 1, 2, 3, 4). If the value is 0, that mean the report is for annual report.
    category_code VARCHAR(255) references map_category_code_bank(category_code),
    data float -- The value of the recorded category (in Million VND)
);

-- Table non_bank_financial_report: Financial report of corporation. Same structure as `bank_financial_report`
CREATE TABLE non_bank_financial_report(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    category_code VARCHAR(255) references map_category_code_non_bank(category_code),
    data float
);

-- Table securities_financial_report: Financial report of securities firms. Same structure as `bank_financial_report`
CREATE TABLE securities_financial_report(
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    category_code VARCHAR(255) references map_category_code_securities(category_code),
    data float
);

-- Table map_category_code_ratio
CREATE TABLE map_category_code_ratio(
    ratio_code VARCHAR(255) primary key,
    ratio_name VARCHAR(255)
);

-- Table financial_ratio: This table will have pre-calculated common Financial Ratio such as ROA, ROE, FCF, etc
-- Same structure as `bank_financial_report`
CREATE TABLE financial_ratio(
    ratio_code VARCHAR(255) references map_category_code_ratio(ratio_code),
    stock_code VARCHAR(255) references company_info(stock_code),
    year int,
    quarter int,
    data float
)

```

<example>
Here is the example of the difficulty evaluation from Spider 1.0 dataset:

### Easy (1)
What is the number of cars with more than 4 cylinders?

SELECT COUNT(*)
FROM cars_data
WHERE cylinders > 4;

### Medium (2)
For each stadium, how many concerts are there?

SELECT T2.name, COUNT(*)
FROM concert AS T1 JOIN stadium AS T2
ON T1.stadium_id = T2.stadium_id
GROUP BY T1.stadium_id;

### Hard (3)
Which countries in Europe have at least 3 car manufacturers?

SELECT T1.country_name
FROM countries AS T1 JOIN continents AS T2
ON T1.continent = T2.cont_id
JOIN car_makers AS T3
ON T1.country_id = T3.country
WHERE T2.continent = 'Europe'
GROUP BY T1.country_name
HAVING COUNT(*) >= 3;

### Extra Hard (4)
What is the average life expectancy in the countries where English is not the official language?

SELECT AVG(life_expectancy)
FROM country
WHERE name NOT IN
(SELECT T1.name
 FROM country AS T1 JOIN country_language AS T2
 ON T1.code = T2.country_code
 WHERE T2.language = 'English'
 AND T2.is_official = 'T');

</example>

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

Based on the evaluation of Spider 1.0 dataset, please evaluate the difficulty of the question and the SQL query. The difficulty should be in the range of 1 to 4, where 1 is the easiest and 4 is the hardest.
Return the difficulty in JSON format.
```json
{{
    "difficulty": 1
}}
```

</task>

"""
    messages = [
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
    time.sleep(5)
    return qa

def parallel_evaluate_difficult(*args):
    llm = ChatGPT('gpt-4o')

    return evaluate_difficult(llm, *args)

def evaluate_difficult_qa():
        
    with open('gpt-4o-generated-v2-scored-pass.json') as f:
        data = json.load(f)
    
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_task = {executor.submit(parallel_evaluate_difficult, qa): qa for qa in data}
        for future in as_completed(future_to_task):
            qa = future_to_task[future]
            results.append(qa)
            
    return results


if __name__ == '__main__':
    results = evaluate_difficult_qa()
    with open('gpt-4o-generated-v2-scored.json', 'w') as f:
        json.dump(results, f, indent=4)