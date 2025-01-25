from pydantic import BaseModel, ConfigDict, SkipValiation, Field
from typing import List, Dict, Any, Optional
from collections import deque

import numpy as np
import random
import copy

import os
import sys 
sys.path.append('..')

current_dir = os.path.dirname(__file__)

from text2sql import Text2SQL # Base solver
import text2sql_utils as utils

from llm.llm_utils import get_code_from_text_response, get_json_from_text_response



class SQLReasoningNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task: str 
    content: str # The reasoning content
    critique: str = ''
    error: List[str] = []
    children: List['SQLReasoningNode'] = []
    parent: Optional['SQLReasoningNode'] = None
    rank: int = 0
    reward: List[float] = []
    Q: float = -1
    N: int = 0
    max_child: int = 2

    solver: Text2SQL
    sql: List = []
    table: List[utils.Table] = []

    def __str__(self):
        return self.content
    
    def add_child(self, node: 'SQLReasoningNode') -> None:
        
        # Update rank
        node.rank = self.rank + 1
        
        # Add node to the tree
        node.parent = self
        self.children.append(node)

    def add_critique(self, critique: str) -> None:
        # Update critique to current reasoning node
        self.critique = critique

    def get_critique(self) -> str:
        return self.critique
    
    def add_error(self, error: str) -> None:
        # Update error to current reasoning node
        self.error = error

    def get_error(self) -> str:
        return self.error
    
    def add_table(self, table: utils.Table|List[utils.Table]) -> None:
        if isinstance(table, list):
            self.table.extend(table)
        else:
            self.table.append(table)

    def get_table(self) -> List[utils.Table]:
        return self.table
    
    def add_reward(self, reward: float) -> None:
        # Update reward to current reasoning node
        self.reward.append(reward)

        avg_reward = np.mean(self.reward)
        min_reward = np.min(self.reward)

        q_value = (min_reward + avg_reward)/2

        self.update_Q(q_value)

    def update_Q(self, Q: float) -> None:
        # Update Q value
        self.Q = Q

    def update_N(self) -> None:
        # Update N value
        self.N +=1

    def local_solution(self) -> str:
        # Get the local solution of the node
        content = f"<reasoning>\n\n {self.content} \n\n</reasoning>"
        
        # Add table result
        if len(self.table)>0:
            content += f"\n\n<table>\n\n {utils.table_to_string(self.table)} \n\n</table>" 

        # Add error SQL
        if len(self.error)>0:
            content += '\n\n<error>\n\n'
            for err in self.error:
                content += f"{err}\n"
            content += '\n\n</error>'

        # Add critique
        if self.critique != '':
            content += f"\n\n<critique>\n\n {self.critique} \n\n</critique>"

        # Add reward
        if self.Q != -1:
            content += f"\n\n<reward>\n\n {self.Q} \n\n</reward>"

        return content

    def get_solution(self, full_solution = False, denote = 'Solution') -> str:
        # Traverse the tree to get the full solution
        answer = f"<task>\n\n{self.task}\n\n</task>\n\n"

        if full_solution:

            reasoning_chain = []
            node = copy.deepcopy(self)
            while node.parent:
                reasoning_chain.append(node)
                node = node.parent

            reasoning_chain = reasoning_chain[::-1]
            
            for i, node in enumerate(reasoning_chain):
                if i == len(reasoning_chain) - 1:
                    tag = f'Previous {denote}'
                else:
                    tag = f'\n\n{i+1} {denote}\n\n'
                
                answer += tag
                answer += node.local_solution()

        else:
            answer += self.local_solution()
        
        return answer
                

    def is_fully_expanded(self) -> bool:
        # Check if the node reach the limit of exploration
        if len(self.children) >= self.max_child:
            return True
        
        # If Q value of the child > current node
        flag = False
        for child in self.children:
            if child.Q > self.Q:
                flag = True
                break

        return flag




class TreeOfThought(BaseModel):
    root: SQLReasoningNode = SQLReasoningNode(task = "", content="")
    
    reward_model: Any = Field(defalt=None)
    critique_model: Any = Field(default=None)
    refine_model: Any = Field(default=None)

    question: str = ''
    verbose: bool = False

    def reward(self, node: SQLReasoningNode) -> float:
        raise NotImplementedError('Not Implement Reward model')
    
    def critique(self, node: SQLReasoningNode) -> None:
        raise NotImplementedError('Not Implement Critique model')

    def refine(self, node: SQLReasoningNode) -> None:
        # Kinda like actor in actor-critic
        raise NotImplementedError('Not Implement Refine (Actor) model')

    @staticmethod
    def UCB(node: SQLReasoningNode, c = 1, epsilon = 1e-6):
        if node.parent is not None:
            ucb = node.Q + c * np.sqrt(2 * np.log(node.parent.N +1) / (node.N + epsilon))
            return ucb
        else:
            return 10000


    def backward(self, node: SQLReasoningNode) -> None:
        
        # Backward of MCTS is get the maxQ of the children, and update the Q value based on it
        while node.parent is not None: # Travel back to root
            parent = node.parent
            parent.update_N()

            # Max Q value of child node
            maxQ = max([child.Q for child in parent.children])

            # Update the Q value of parent node
            parent.Q = (parent.Q + maxQ) / 2

            # Travel
            node = parent


    def select_node(self, search_policy: str = 'sampling') -> SQLReasoningNode:
        
        candidate = []
        bfs_node = deque([self.root])

        # Do BFS to find the node that is not fully expanded
        while bfs_node:
            node = bfs_node.popleft()
            if not node.is_fully_expanded():
                candidate.append(node)
            bfs_node.extend(node.children)

        if not candidate:
            return self.root
        
        # Choose the next node based on the search policy

        # Greedy: Choose the node with the highest UCB value
        if search_policy.lower() == 'greedy':
            return max(candidate, key=lambda x: self.UCB(x))

        # Sampling: Treet the UCB value as the probability to choose the node
        elif search_policy.lower() == 'sampling':
            ucbs = [self.UCB(node) for node in candidate]
            choices = random.choices(candidate, weights=ucbs, k=1)[0]
            return candidate[choices]
        
        else:
            raise ValueError('Invalid search policy')

# Solution based on Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report
# https://arxiv.org/pdf/2406.07394

class SelfRefine_TOT(TreeOfThought):

    solver: Text2SQL
    def reward(self, node: SQLReasoningNode, full_solution = False) -> float:
        
        # Using LLM to grade the result
        system_prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/system/reward.txt'))
        prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/content/reward.txt'))

        # Get the partial answer
        solution = node.get_solution(full_solution = full_solution)

        prompt = prompt.format(question = self.question, solution = solution)

        # Grade with LLM
        messages = [
            {
                'role' : 'system',
                'content' : system_prompt
            },
            {
                'role' : 'user',
                'content' : prompt
            }
        ]

        response = self.reward_model(messages)
        if self.verbose:
            print(f"    Reward:\n\n{response}\n\n====================")

        try:
            score = get_json_from_text_response(response, new_method=True)['score']
        except Exception as e:
            print(e)
            score = 0

        return score

    def critique(self, node: SQLReasoningNode) -> None:
        
        # Using LLM to critique the response
        system_prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/system/reward.txt'))
        prompt = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/content/reward.txt'))

        solution = node.get_solution(False)

        prompt = prompt.format(question = self.question, solution = solution)
        
        # Critique with LLM
        messages = [
            {
                'role' : 'system',
                'content' : system_prompt
            },
            {
                'role' : 'user',
                'content' : prompt
            }
        ]

        critique = self.critique_model(messages)

        if self.verbose:
            print(f"    Critique:\n\n{critique}\n\n====================")
    
        # Add critque to current node
        node.add_critique(critique)

    def _solve_a_task(self, solver: Text2SQL, task: str) -> SQLReasoningNode:
        # Solve the task and get the result
        history_log, error_messages, tables = solver.solve(task)
        content, _, _ = solver.refine_error_correction(error_messages, tables)
        
        # Add refine to current node
        new_node = SQLReasoningNode(task = task, content=content, solver=solver)
        new_node.add_error(error_messages)
        new_node.add_table(tables)

        return new_node

    def next_step(self, node: SQLReasoningNode) -> SQLReasoningNode:
        # Kinda like actor in actor-critic

        # Get the critique
        critque = node.critique


        # New task to refine
        refine_task = utils.read_file_without_comments(os.path.join(current_dir, 'prompt/content/refine.txt'))
        refine_task = refine_task.format(critique = critque)


        # Copy the current solver
        child_solver = copy.deepcopy(node.solver)

        # Solve the task and get the result
        new_node = self._solve_a_task(child_solver, refine_task)

        node.add_child(new_node)

        return new_node


    def clear(self):
        self.root = SQLReasoningNode(task ='', content="")
        self.solver.reset()

    def start(self, question: str, weak_answer: bool = False) -> SQLReasoningNode:

        # Clear the tree
        self.clear()

        # Set the question
        self.question = question
        
        if weak_answer:
            # Currently, we don't have a weak answer
            node = SQLReasoningNode(task = question, content="I dont't know", solver=copy.deepcopy(self.solver))
        else:
            node = SQLReasoningNode(task = question, content="I don't know", solver=copy.deepcopy(self.solver))

        node.update_Q(0)

    def rollout_one(self, node: SQLReasoningNode) -> SQLReasoningNode:

        self.critique(node)
        new_node = self.next_step(node) # Refine the solution
        
        r = self.reward(new_node)
        new_node.add_reward(r)

        return new_node

    def mcts(self, question: str, max_iter: int = 10, verbose: bool = False, weak_answer : bool = False, max_score = 10) -> SQLReasoningNode:
        
        self.root = self.start(question, weak_answer)

        iteration = 0
        max_Q = 0

        while iteration < max_iter:
            # Rollout one step
            selected_node = self.select_node()
            new_node = self.rollout_one(selected_node)
            self.backward(new_node)

            # Update max Q value
            if new_node.Q > max_Q:
                max_Q = new_node.Q

            iteration += 1
            if verbose:
                print(f"iteration: {iteration}, Q: {new_node.Q}")

            # Break condition
            if max(new_node.reward) == max_score: # If the score is max, break
                break
        
        return self.select_node()