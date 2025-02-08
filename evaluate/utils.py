import os 
import json

import sys 

sys.path.append('..')

from agent.const import (
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG,
    TEXT2SQL_MEDIUM_GEMINI_CONFIG,
    TEXT2SQL_GEMINI_PRO_CONFIG,
    TEXT2SQL_THINKING_GEMINI_CONFIG,
    TEXT2SQL_4O_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL,
    HORIZONTAL_PROMPT_BASE,
    HORIZONTAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI,
)


def get_prompt_config(template):
    if 'openai' in template:
        print("===== Using openai =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI
    elif 'simplify' in template:
        print("===== Using simplify =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY
    else:
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL
    return prompt_config



def get_text2sql_config(llm_name):
    if 'gemini' in llm_name:

        if 'thinking' in llm_name:
            return TEXT2SQL_THINKING_GEMINI_CONFIG
        if 'gemini-pro' in llm_name:
            return TEXT2SQL_GEMINI_PRO_CONFIG
        return TEXT2SQL_FAST_GEMINI_CONFIG
    
    if 'gpt-4o' in llm_name:
        if 'mini' not in llm_name:
            return TEXT2SQL_4O_CONFIG
        return TEXT2SQL_FAST_OPENAI_CONFIG

    if 'deepseek-chat' in llm_name:
        return TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG

    else:
        config = TEXT2SQL_FAST_GEMINI_CONFIG
        config['sql_llm'] = llm_name    
        return config



def append_jsonl_to_file(json_obj, file_path):
    with open(file_path, 'a') as f:
        json.dump(json_obj, f)
        f.write('\n')
        

def get_available_path(path):
    if not os.path.exists(path):
        return path
    else:
        i = 1
        file_type = path.split('.')[-1]
        path = path.replace('.' + file_type, '')

        while True:
            new_path = path + '_' + str(i) + '.' + file_type
            if not os.path.exists(new_path):
                return new_path
            i += 1

        