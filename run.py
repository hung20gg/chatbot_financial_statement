from ETL import setup_db, setup_db_openai
from agent import Chatbot, Text2SQL
from agent.const import (
    Config,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    TEXT2SQL_MEDIUM_GEMINI_CONFIG,
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_SWEET_SPOT_CONFIG
)

from langchain_huggingface import HuggingFaceEmbeddings




if __name__ == "__main__":
    model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': 'cuda'})
    db = setup_db(model)
    # db = setup_db_openai(multi_thread=False)
    config = Config(**GEMINI_FAST_CONFIG)
    text2sql_config = Text2SQLConfig(**TEXT2SQL_FASTEST_CONFIG)
    
    text2sql = Text2SQL(text2sql_config, db)
    
    db.search_return_df('total assets', 2, 'bank')
    prompt = "Compare the Return on Assets (ROA) and Return on Equity (ROE) of Vinamilk and Masan Group for the fiscal year 2023.  Additionally, provide the total assets and total equity for both companies for the same period."
    his, err, tab = text2sql.solve(prompt)
    
    for t in tab:
        print(t)
    
    print("Hello World!")