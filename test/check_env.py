import os
import pandas as pd

env_vars = dict(os.environ)
for key, value in env_vars.items():
    print(f"{key}: {value}")
    
if 'DEEPINFRA_TOKEN' in os.environ:
    del os.environ['DEEPINFRA_TOKEN']
if 'GENAI_API_KEY' in os.environ:
    del os.environ['GENAI_API_KEY']
    
    # if 'OPENAI_API_KEY' in os.environ:
    #     print()
    
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']