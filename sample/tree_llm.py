from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
import re
import copy

def next_step(llm, task, partial_response):
    with open('prompt/tree/system_prompt_next_step.txt', 'r') as f:
        system_prompt = f.read()
        
    prompt = f"""
You are tasked to solve the following problem:
{task}    

Here is the partial solution that you have generated:
<thought>
{partial_response}
</thought>

Think step-by-step and return the next step to solve the problem. If there are multiple possible next step, you can provide all of them, at most 3 different methods.
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
    
    response = llm(messages)
    
    messages.append({
        "role": "assistant",
        "content": response
    })
    
    # Return the next step in JSON format
    
    force_json = """
    Return your previous answer in JSON format. 
    
    <formatting_example>
    ```json
    {
        "next_step": ["SELECT * FROM table_name WHERE condition == \\"text\\";"]
    }
    ```
    </formatting_example>
    """
    
    messages.append({
        "role": "user",
        "content": force_json
    })
    return get_json_from_text_response(response, new_method=True)

def reward_llm(llm, task, partial_response, next_step):
    with open('prompt/tree/system_prompt_reward.txt', 'r') as f:
        system_prompt = f.read()
        
    prompt = f"""
    This is the task that someone have been working on:
    {task}
    
    Notice that this might just be a partial solution. Here is the partial solution:
    
    <thought>
    {partial_response}
    </thought>
    """

def get_reasoning_chain(node):
    reason_chain = []
    while node.parent is not None:
        reason_chain.append(node.content)
        node = node.parent
    
    reason_chain.append(node.content)
    reason_chain.reverse()
    
    text = ""
    for i, reason in enumerate(reason_chain):
        text += "Step {}: {}\n".format(i + 1, reason)
    return text

class ReasoningNode:
    def __init__(self, content, score = 0.5):
        self.content = content 
        self.parent = None 
        self.children = []
        self.rank = 0
        self.score = score
        
    def add(self, node):
        node.rank = self.rank + 1
        self.children.append(node)
        node.parent = self
        
        
