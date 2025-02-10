import os 
import json

import sys 

sys.path.append('..')

from agent.const import (
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_FAST_GEMINI_CONFIG,
    TEXT2SQL_FAST_OPENAI_CONFIG,
    TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG,
    TEXT2SQL_FAST_SQL_OPENAI_CONFIG,
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
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SHORT
)


def get_prompt_config(template):
    if 'openai' in template:
        print("===== Using openai =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI
    elif 'simplify' in template:
        print("===== Using simplify =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY
    elif 'short' in template:
        print("===== Using short =====")
        prompt_config = FIIN_VERTICAL_PROMPT_UNIVERSAL_SHORT
    else:
        print("===== Using universal =====")
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
        return TEXT2SQL_FAST_SQL_OPENAI_CONFIG

    if 'deepseek-chat' in llm_name:
        return TEXT2SQL_DEEPSEEK_V3_FAST_CONFIG

    else:
        config = TEXT2SQL_FAST_GEMINI_CONFIG
        config['sql_llm'] = llm_name    
        return config


def get_avaliable_questions(input_path, reference_paths = [], max_questions = 2000):
    selected_questions = []
    done_ids = set()
    
    # Read reference file and get all done ids
    if isinstance(reference_paths, str):
        path = reference_paths.split(',')
        reference_paths = path
        
    for reference_path in reference_paths:
        if os.path.exists(reference_path):
            print(f"Reading reference file: {reference_path}")

            if reference_path.endswith('.jsonl'):
                with open(reference_path, 'r') as f:
                    for line in f:
                        try:
                            msg_obj = json.loads(line)
                            done_ids.add(msg_obj['ids'])
                        except:
                            print(f"Error reading line:")
                            
            elif reference_path.endswith('.json'):
                with open(reference_path, 'r') as f:
                    questions = json.load(f)
                    for question in questions:
                        done_ids.add(question['ids'])
        

    # Read input file and get all questions
    if input_path.endswith('.jsonl'): 
        print(f"Reading JSONL input file: {input_path}")
        with open(input_path, 'r') as f:
            for line in f:
                question = json.loads(line)
                if question['ids'] not in done_ids:
                    selected_questions.append(question)

    elif input_path.endswith('.json'):
        print(f"Reading JSON input file: {input_path}")
        with open(input_path, 'r') as f:
            questions = json.load(f)
            for question in questions:
                if question['ids'] not in done_ids:
                    selected_questions.append(question)
    else:
        raise ValueError("Input file must be json or jsonl")
                                     

    print(f"Total questions: {len(selected_questions)}")
    return selected_questions


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

        