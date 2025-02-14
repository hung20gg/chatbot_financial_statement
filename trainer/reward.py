import re

import sys 
sys.path.append('..')

from initialize import initialize_text2sql
from agent.const import (
    FIIN_VERTICAL_PROMPT_UNIVERSAL_SIMPLIFY,
    FIIN_VERTICAL_PROMPT_UNIVERSAL_OPENAI,

    TEXT2SQL_FAST_GEMINI_CONFIG
)

from ETL.dbmanager.setup import (
    setup_db,
    DBConfig,
    TEI_VERTICAL_UNIVERSAL_CONFIG
)

from agent.text2sql_utils import get_llm_wrapper, TIR_reasoning, table_to_markdown
from llm.llm_utils import get_json_from_text_response
from concurrent.futures import ThreadPoolExecutor


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_think(text: str) -> str:
    answer = text.split("<think>")[-1]
    answer = answer.split("</think>")[0]
    return answer.strip()

def extract_xml_error(text: str) -> str:
    answer = text.split("<error>")[-1]
    answer = answer.split("</error>")[0]
    return answer.strip()

def sql_execution(contents: list[str]) -> list[str]: 
    results = []

    db = setup_db(DBConfig(**TEI_VERTICAL_UNIVERSAL_CONFIG))

    for content in contents:
        errors, tables = TIR_reasoning(content, db=db)

        msg = ""
        if len(errors) > 0:
            msg +="<error>\n\n"
            for error in errors:
                msg += f"{error}\n"
            msg += "</error>\n\n"

        if len(tables) > 0:
            msg += "<result>\n\n{table_to_markdown(tables)}\n\n<result>"

        results.append(msg)
    return results

def _llm_as_a_judge(task: str, response: str, ground_truth: str) -> float:
    
    prompt = ""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    llm = get_llm_wrapper('gemini-2.0-flash')

    response = llm(messages)

    try:
        reward = get_json_from_text_response(response)['score']
    except :
        reward = 0.0

    return reward

def llm_as_a_judge(tasks: list[str], responses: list[str], ground_truths: list[str]) -> float:
    with ThreadPoolExecutor() as executor:
        future = [executor.submit(_llm_as_a_judge, task, response, ground_truth) for task, response, ground_truth in zip(tasks, responses, ground_truths)]
        return future.result()
    
def sql_reward_func(completions, tasks, ground_truths, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    sql_responses = sql_execution(responses)
    errors = [extract_xml_error(r) for r in sql_responses]
    sql_rewards = llm_as_a_judge(tasks, sql_responses, ground_truths)
    
    rewards = []
    for error, sql_reward in zip(errors, sql_rewards):
        if error:
            rewards.append(0.0)
        else:
            rewards.append(sql_reward)
    return rewards