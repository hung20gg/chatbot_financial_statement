from pydantic import BaseModel
from .. import text2sql_utils as utils

import os
current_dir = os.path.dirname(__file__)

class PromptConfig(BaseModel):
    BREAKDOWN_NOTE_PROMPT: str 
    OPENAI_SEEK_DATABASE_PROMPT: str 
    GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT: str 
    BRANCH_REASONING_PROMPT: str 
    REASONING_TEXT2SQL_PROMPT: str 
    BRANCH_REASONING_TEXT2SQL_PROMPT: str
    
    
VERTICAL_PROMPT_BASE = {
    "BREAKDOWN_NOTE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/base/breakdown_note.txt')),
    "OPENAI_SEEK_DATABASE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/base/openai_seek_database.txt'), start=['//']),
    "GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/base/get_stock_code_and_suitable_row.txt'), start=['//']),
    "BRANCH_REASONING_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/base/branch_reasoning.txt'), start=['//']),
    "REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/base/reasoning_text2sql.txt'), start=['//']),
    "BRANCH_REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/base/branch_reasoning_text2sql.txt'), start=['//'])
}

VERTICAL_PROMPT_UNIVERSAL = {
    "BREAKDOWN_NOTE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/universal/breakdown_note.txt')),
    "OPENAI_SEEK_DATABASE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/universal/openai_seek_database.txt'), start=['//']),
    "GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/universal/get_stock_code_and_suitable_row.txt'), start=['//']),
    "BRANCH_REASONING_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/universal/branch_reasoning.txt'), start=['//']),
    "REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/universal/reasoning_text2sql.txt'), start=['//']),
    "BRANCH_REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'vertical/universal/branch_reasoning_text2sql.txt'), start=['//'])    
}

HORIZONTAL_PROMPT_BASE = {
    "BREAKDOWN_NOTE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/base/breakdown_note.txt')),
    "OPENAI_SEEK_DATABASE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/base/openai_seek_database.txt'), start=['//']),
    "GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/base/get_stock_code_and_suitable_row.txt'), start=['//']),
    "BRANCH_REASONING_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/base/branch_reasoning.txt'), start=['//']),
    "REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/base/reasoning_text2sql.txt'), start=['//']),
    "BRANCH_REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/base/branch_reasoning_text2sql.txt'), start=['//'])
}

HORIZONTAL_PROMPT_UNIVERSAL = {
    "BREAKDOWN_NOTE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/universal/breakdown_note.txt')),
    "OPENAI_SEEK_DATABASE_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/universal/openai_seek_database.txt'), start=['//']),
    "GET_STOCK_CODE_AND_SUITABLE_ROW_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/universal/get_stock_code_and_suitable_row.txt'), start=['//']),
    "BRANCH_REASONING_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/universal/branch_reasoning.txt'), start=['//']),
    "REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/universal/reasoning_text2sql.txt'), start=['//']),
    "BRANCH_REASONING_TEXT2SQL_PROMPT": utils.read_file_without_comments(os.path.join(current_dir, 'horizontal/universal/branch_reasoning_text2sql.txt'), start=['//'])   
}