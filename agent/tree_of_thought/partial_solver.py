from base import SelfRefine_TOT, SQLReasoningNode
from typing import List
import copy

import os
import sys 
sys.path.append('..')

current_dir = os.path.dirname(__file__)

import text2sql_utils as utils
from llm.llm_utils import get_code_from_text_response, get_json_from_text_response

# Solution based on ReST-MCTS∗: LLM Self-Training via Process Reward Guided Tree Search

def get_solution_from_text_response(response: str) -> List:
    pass

class Partial_SelfRefine_TOT(SelfRefine_TOT):
    
    def is_terminal(self, node: SQLReasoningNode) -> bool:

        system_prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/system/reward.txt'))
        prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/content/reward.txt'))


        solution = node.get_solution(full_solution=True)

        prompt = prompt.format(solution=solution)

        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Temporary using the critique model to check if the full solution is meet
        response = self.critique_model(message)

        if self.verbose:
            print('    Check termination')
        try:
            flag = get_json_from_text_response(response)['is_full_solution']
        except Exception as e:
            print('Error in is_terminal', e)
            flag = False

        return flag
    
    def prm(self, node: SQLReasoningNode) -> float:
        return self.reward(node, full_solution=True)

    def start(self, question: str, weak_answer: bool = False) -> SQLReasoningNode:
        
        self.clear()

        self.question = question

        if weak_answer:
            # Check the solver type:
            system_prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/system/reward.txt'))
            prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/content/reward.txt'))

            prompt = prompt.format(question=question)

            message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Generate a weak solution
            response = self.refine_model(message)
            
            try:
                tasks = get_solution_from_text_response(response)
                previous_node = SQLReasoningNode(content = "I don't know", solver = copy.deepcopy(self.solver), critique = "No solution available")

                # Solve the tasks
                for task in tasks:
                    child_solver = copy.deepcopy(previous_node.solver)
                    new_node = self._solve_a_task(solver=child_solver, task=task)
                    previous_node.add_child(new_node)
                    previous_node = new_node

                # Remove the first node
                previous_node = previous_node.children[0]
                return previous_node
            except Exception as e:
                print('Error in start', e)
                return SQLReasoningNode(content = "I don't know", solver = copy.deepcopy(self.solver), critique = "No solution available")

            


        else:
            new_node = SQLReasoningNode(content = "I don't know", solver = copy.deepcopy(self.solver), critique = "No solution available")
            new_node.add_critique("No solution available")
            
        return new_node