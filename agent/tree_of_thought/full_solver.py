import os
import sys 
sys.path.append('..')

from base import SelfRefine_TOT, SQLReasoningNode
from text2sql import Text2SQL # Base solver
import text2sql_utils as utils

current_dir = os.path.dirname(__file__)

from llm.llm_utils import get_code_from_text_response, get_json_from_text_response

# Solution based on Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report
# https://arxiv.org/pdf/2406.07394
# However, we make some modifications to the original algorithm, making it more like Actor-Critic
# Changes focus on rolling out the tree and get new solutions

# Each node now has a content (solution) and a critic (cooment) when initializing
# RM is based on the critic of the node (A-C)

class Full_SelfRefine_TOT(SelfRefine_TOT):
    
    def start(self, question: str, weak_answer: bool = False) -> SQLReasoningNode:
        new_node = super().start(question, weak_answer)
        new_node.add_critique("No solution available")
        return new_node

    def rollout_one(self, node: SQLReasoningNode) -> SQLReasoningNode:
        
        new_node = self.next_step(node)
        self.critique(new_node)

        r = self.reward(new_node)
        new_node.add_reward(r)
        return new_node