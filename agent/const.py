from pydantic import BaseModel
from . import text2sql_utils as utils
import os
current_dir = os.path.dirname(__file__)


class Config(BaseModel):
    llm: str
    routing_llm: str
    summary_every: int = -1
    
class Text2SQLConfig(BaseModel):
    llm: str
    sql_llm: str
    reasoning: bool = True
    branch_reasoning: bool = True
    company_top_k: int = 2
    sql_example_top_k: int = 2
    account_top_k: int = 5
    verbose: bool = False
    get_all_acount: bool = False    
    
GEMINI_FAST_CONFIG = {
    "llm": 'gemini-1.5-flash-002',
    "routing_llm": 'gemini-1.5-flash-002',
    "summary_every": -1
}

GEMINI_BEST_CONFIG = {
    "llm": 'gemini-1.5-pro-002',
    "routing_llm": 'gemini-1.5-flash-002',
    "summary_every": -1
}

OPENAI_FAST_CONFIG = {
    "llm": 'gpt-4o-mini',
    "routing_llm": 'gpt-4o-mini',
    "summary_every": -1
}

OPENAI_BEST_CONFIG = {
    "llm": 'gpt-4o',
    "routing_llm": 'gpt-4o-mini',
    "summary_every": -1
} 

    
BEST_CONFIG = {
    "llm": 'gpt-4o',
    "sql_llm": 'gpt-4o',
    "self_debug": True,
    "reasoning": True,
    "branch_reasoning": True,
    "company_top_k": 2,
    "sql_example_top_k": 2,
    "account_top_k": 5,
    "verbose": False,
    'get_all_acount': True
}

MEDIUM_OPENAI_CONFIG = {
    "llm": 'gpt-4o-mini',
    "sql_llm": 'gpt-4o-mini',
    "reasoning": False,
    "branch_reasoning": True,
    "company_top_k": 2,
    "sql_example_top_k": 2,
    "account_top_k": 4,
    "verbose": False,
    'get_all_acount': False
}

MEDIUM_GEMINI_CONFIG = {
    "llm": 'gemini-1.5-flash-002',
    "sql_llm": 'gemini-1.5-flash-002',
    "reasoning": False,
    "branch_reasoning": True,
    "company_top_k": 2,
    "sql_example_top_k": 2,
    "account_top_k": 4,
    "verbose": False,
    'get_all_acount': False
}

FASTEST_CONFIG = {
    "llm": 'gemini-1.5-flash-8b',
    "sql_llm": 'gemini-1.5-flash-002',
    "reasoning": False,
    "branch_reasoning": False,
    "company_top_k": 2,
    "sql_example_top_k": 2,
    "account_top_k": 4,
    "verbose": False,
    'get_all_acount': False
}


BREAKDOWN_NOTE_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/breakdown_note.txt'))

OPENAI_SEEK_DATABASE_PROMPT  = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/openai_seek_database.txt'))
    
GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT  = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/get_stock_code_and_suitable_row.txt'))

BRANCH_REASONING_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/branch_reasoning.txt'))
    
REASONING_TEXT2SQL_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/reasoning_text2sql.txt'))