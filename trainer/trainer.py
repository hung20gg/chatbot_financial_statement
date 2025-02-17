from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

from reward import format_reward_func, sql_reward_func


def get_grpo_config():
    pass

def get_lora_config():
    pass