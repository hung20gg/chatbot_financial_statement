from .base import BaseAgent
from pydantic import BaseModel
import text2sql_utils as utils
import sys 
sys.path.append('..')
from ETL.hub import DBHUB
from llm.llm.abstract import LLM
from llm.llm_utils import get_json_from_text_response, get_code_from_text_response


class Config(BaseModel):
    llm: LLM
    sql_llm: LLM
    reasoning = True
    branch_reasoning = True
    company_top_k = 2
    sql_example_top_k = 2
    account_top_k = 5
    verbose = False
    

class Text2SQL(BaseAgent):
    def __init__(self, config: Config, db: DBHUB):
        super().__init__(config)
        
        self.db = db
        self.llm = config.llm
        if hasattr(config, 'sql_llm'):
            self.sql_llm = config.sql_llm
        else:
            self.sql_llm = config.llm
            
        self.reasoning = config.reasoning
        self.branch_reasoning = config.branch_reasoning
        
        self.history = []
        self.llm_responses = []
        
    def get_stock_code_based_on_company_name(self, task, get_industry=False):
        
        messages = [
        {
            "role": "user",
            "content": f"""
Extract the company name and/or the industry that positively mentioned based on the given question.
<question>
{task}
</question>
Only return exact the company name mentioned. Do not answer the question.
Return in JSON format. 

```json
{{
    "industry": [],
    "company_name": ["company1"]
}}
```
Return an empty list if no company name is found.
"""}]
        
        response = self.llm(messages)
        if self.config.verbose:
            print("Get stock code based on company name response: ")
            print(response)
            print("====================================")
            
        company_names = get_json_from_text_response(response, new_method=True)['company_name']
        industries = response.get("industry", [])
        
        if get_industry:
            return utils.company_name_to_stock_code(self.db, company_names, top_k=self.config.company_top_k), industries
        return company_names
    
    
    def find_suitable_row_v2(self, text, stock_code = [], format = 'dataframe'):
        system_prompt = """
    You are an expert in analyzing financial reports. You are given 2 database, finacial statements and pre-calculated pre-calculated financial performance ratios.
    """
    
        prompt = f"""
    <thought>
    {text}
    </thought>

    <task>
    Based on given question, analyze and suggest the industry and suitable rows (categories) in the financial statement and/or financial performance ratios that can be used to answer the question.
    Analyze and return the suggested rows' name in JSON format.
    </task>

    <formatting_example>
    ```json
    {{
        "industry": [],
        "financial_statement_row": [],
        "financial_ratio_row": []
    }}
    ```
    </formatting_example>
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
    
        response = self.llm(messages)
        
        if self.config.verbose:
            print("Find suitable row response: ")
            print(response)
            print("====================================")
            
        response = get_json_from_text_response(response, new_method=True)
        
        industry = response.get("industry", [])
        financial_statement_row = response.get("financial_statement_row", [])
        financial_ratio_row = response.get("financial_ratio_row", [])