from pydantic import BaseModel
from . import text2sql_utils as utils
import os
current_dir = os.path.dirname(__file__)
from .prompt.prompt_controller import *


class Config(BaseModel):
    llm: str
    
    
class ChatConfig(Config):
    routing_llm: str
    summary_every: int = -1
    get_task: bool = True
    
    
class Text2SQLConfig(Config):
    sql_llm: str
    reasoning: bool = True
    branch_reasoning: bool = False
    company_top_k: int = 2
    sql_example_top_k: int = 3
    account_top_k: int = 6
    verbose: bool = False
    get_all_acount: bool = False    
    self_debug: bool = False
    
GEMINI_FAST_CONFIG = {
    "llm": 'gemini-2.0-flash',
    "routing_llm": 'gemini-2.0-flash',
    "summary_every": -1,
    "get_task": True
}

GEMINI_FAST_CONFIG_V2 = {
    "llm": 'gemini-2.0-flash',
    "routing_llm": 'gemini-2.0-flash',
    "summary_every": -1,
    "get_task": True
}

GEMINI_EXP_CONFIG = {
    "llm": 'gemini-2.0-flash-exp',
    "routing_llm": 'gemini-2.0-flash-exp',
    "summary_every": -1,
    "get_task": True
}


INBETWEEN_CHAT_CONFIG = {
    "llm": 'gpt-4o-mini',
    "routing_llm": 'gemini-2.0-flash-002',
    "summary_every": -1,
    "get_task": True
}

GPT4O_MINI_CONFIG = {
    "llm": 'gpt-4o-mini',
    "routing_llm": 'gpt-4o-mini',
    "summary_every": -1
}

GPT4O_CONFIG = {
    "llm": 'gpt-4o',
    "routing_llm": 'gpt-4o-mini',
    "summary_every": -1
}

GEMINI_BEST_CONFIG = {
    "llm": 'gemini-1.5-pro-002',
    "routing_llm": 'gemini-2.0-flash',
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

    
TEXT2SQL_BEST_CONFIG = {
    "llm": 'gpt-4o',
    "sql_llm": 'gpt-4o',
    "self_debug": True,
    "reasoning": False,
    "branch_reasoning": True,
    "company_top_k": 2,
    "sql_example_top_k": 3,
    "account_top_k": 8,
    "verbose": False,
    'get_all_acount': True,
    'self_debug': True
}

TEXT2SQL_4O_CONFIG = {
    "llm": 'gpt-4o',
    "sql_llm": 'gpt-4o',
    "self_debug": True,
    "reasoning": False,

    "account_top_k": 8,
    'get_all_acount': True,
    'self_debug': True
}

TEXT2SQL_GEMINI_PRO_CONFIG = {
    "llm": 'gemini-1.5-pro',
    "sql_llm": 'gemini-1.5-pro',
    "self_debug": True,

    "account_top_k": 7,
    'get_all_acount': True,
    'self_debug': True
}

TEXT2SQL_GEMINI_PRO_EXP_CONFIG = {
    "llm": 'gemini-2.0-pro-exp-02-05',
    "sql_llm": 'gemini-2.0-pro-exp-02-05',
    "self_debug": True,

    "account_top_k": 7,
    'get_all_acount': True,
    'self_debug': True
}


TEXT2SQL_SWEET_SPOT_CONFIG = {
    "llm": 'gemini-1.5-flash-8b',
    "sql_llm": 'gpt-4o-mini',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_FAST_OPENAI_CONFIG = {
    "llm": 'gpt-4o-mini',
    "sql_llm": 'gpt-4o-mini',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_FAST_SQL_OPENAI_CONFIG = {
    "llm": 'gemini-2.0-flash-lite-preview-02-05',
    "sql_llm": 'gpt-4o-mini',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_FAST_GEMINI_CONFIG = {
    "llm": 'gemini-2.0-flash-lite-preview-02-05',
    "sql_llm": 'gemini-2.0-flash',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_DEEPSEEK_V3_CONFIG = {
    "llm": 'deepseek-chat',
    "sql_llm": 'deepseek-chat',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG = {
    "llm": 'gemini-2.0-flash',
    "sql_llm": 'deepseek-chat',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': False
}

TEXT2SQL_DEEPSEEK_REASONING_CONFIG = {
    "llm": 'deepseek-chat',
    "sql_llm": 'deepseek-reasoner',
    "reasoning": True,
    "branch_reasoning": False,
    "company_top_k": 2,
    "sql_example_top_k": 4,
    "account_top_k": 8,
    'self_debug': True
}

TEXT2SQL_MEDIUM_OPENAI_CONFIG = {
    "llm": 'gpt-4o-mini',
    "sql_llm": 'gpt-4o-mini',
    "reasoning": True,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_MEDIUM_GEMINI_CONFIG = {
    "llm": 'gemini-1.5-flash',
    "sql_llm": 'gemini-2.0-flash',
    "reasoning": True,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_EXP_GEMINI_CONFIG = {
    "llm": 'gemini-2.0-flash',
    "sql_llm": 'gemini-2.0-pro-exp-02-05',
    "reasoning": True,
    "branch_reasoning": False,

    'self_debug': True
}

TEXT2SQL_THINKING_GEMINI_CONFIG = {
    "llm": 'gemini-2.0-flash',
    'sql_llm': 'gemini-2.0-flash-thinking-exp-01-21',
    "reasoning": False,
    "branch_reasoning": False,
     "max_solution_cache": 5,

    'self_debug': True
}


TEXT2SQL_FASTEST_CONFIG = {
    "llm": 'gemini-1.5-flash-8b',
    "sql_llm": 'gemini-1.5-flash-002',
    "reasoning": False,
    "branch_reasoning": False,

    'self_debug': False
}


## ====== LOCAL CONFIGS ====== ##


TEXT2SQL_QWEN25_CODER_3B_SFT_CONFIG = {
    "llm": 'qwen2.5-coder-3b-sft',
    "sql_llm": 'qwen2.5-coder-3b-sft',
    "reasoning": False,
    "branch_reasoning": False,
    'account_top_k': 4,
    'sql_example_top_k': 1,

    "max_solution_cache": 1,

    "verbose": False,
    'get_all_acount': False,
    'self_debug': True
}

TEXT2SQL_QWEN25_CODER_3B_DPO_CONFIG = {
    "llm": 'qwen2.5-coder-3b-dpo',
    "sql_llm": 'qwen2.5-coder-3b-dpo',
    "reasoning": False,
    "branch_reasoning": False,
    'account_top_k': 4,
    'sql_example_top_k': 1,

    "max_solution_cache": 1,

    "verbose": False,
    'get_all_acount': False,
    'self_debug': True
}


TEXT2SQL_QWEN25_CODER_3B_KTO_CONFIG = {
    "llm": 'qwen2.5-coder-3b-kto',
    "sql_llm": 'qwen2.5-coder-3b-kto',
    "reasoning": False,
    "branch_reasoning": False,
    'account_top_k': 4,
    'sql_example_top_k': 1,

    "max_solution_cache": 1,

    "verbose": False,
    'get_all_acount': False,
    'self_debug': True
}


TEXT2SQL_QWEN25_CODER_1B_SFT_CONFIG = {
    "llm": 'qwen2.5-coder-1.5b-sft',
    "sql_llm": 'qwen2.5-coder-1.5b-sft',
    "reasoning": False,
    "branch_reasoning": False,
    'account_top_k': 4,
    'sql_example_top_k': 1,

    "max_solution_cache": 1,
    "verbose": False,
    'get_all_acount': False,
    'self_debug': True
}


TEXT2SQL_QWEN25_CODER_1B_DPO_CONFIG = {
    "llm": 'qwen2.5-coder-1.5b-dpo',
    "sql_llm": 'qwen2.5-coder-1.5b-dpo',
    "reasoning": False,
    "branch_reasoning": False,
    'account_top_k': 4,
    'sql_example_top_k': 1,

    "max_solution_cache": 1,
    "verbose": False,
    'get_all_acount': False,
    'self_debug': True
}


TEXT2SQL_QWEN25_CODER_1B_KTO_CONFIG = {
    "llm": 'qwen2.5-coder-1.5b-kto',
    "sql_llm": 'qwen2.5-coder-1.5b-kto',
    "reasoning": False,
    "branch_reasoning": False,
    'account_top_k': 4,
    'sql_example_top_k': 1,

    "max_solution_cache": 1,
    "verbose": False,
    'get_all_acount': False,
    'self_debug': True
}




BREAKDOWN_NOTE_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/breakdown_note.txt'))

OPENAI_SEEK_DATABASE_PROMPT  = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/openai_seek_database.txt'), start=['//'])
    
GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT  = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/get_stock_code_and_suitable_row.txt'), start=['//'])

BRANCH_REASONING_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/branch_reasoning.txt'), start=['//'])
    
REASONING_TEXT2SQL_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/reasoning_text2sql.txt'), start=['//'])

BRANCH_REASONING_TEXT2SQL_PROMPT = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/branch_reasoning_text2sql.txt'), start=['//'])