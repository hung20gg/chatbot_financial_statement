import sys 
sys.path.append('..')
import pandas as pd
import numpy as np


from agent.text2sql_utils import prune_unnecessary_data_from_sql, Table


if __name__ == '__main__':
    
    # Test case 1
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['A', 'B', 'C', 'D', 'E'],
        'is_bank': [10, 20, 30, 40, 50],
    }
    
    df = pd.DataFrame(data)
    table = Table(table=df)
    
    messages = [
        {
            "role": "assistant",
            "content": """
                ```sql
                SELECT * FROM df    
                WHERE name in ('A', 'B', 'C')
                ```    
                
                """
        }
    ]
    
    response = prune_unnecessary_data_from_sql(table, messages)
    print(response.table)
    
    