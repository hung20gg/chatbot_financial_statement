# from ETL import setup_db, setup_db_openai
from agent import Chatbot, Text2SQL
from agent.const import (
    ChatConfig,
    Text2SQLConfig,
    GEMINI_FAST_CONFIG,
    TEXT2SQL_MEDIUM_OPENAI_CONFIG,
    TEXT2SQL_MEDIUM_GEMINI_CONFIG,
    TEXT2SQL_FASTEST_CONFIG,
    TEXT2SQL_SWEET_SPOT_CONFIG
)

from agent.prompt.prompt_controller import (
    PromptConfig, 
    VERTICAL_PROMPT_BASE, 
    VERTICAL_PROMPT_UNIVERSAL
)

from ETL.dbmanager.setup import (
    DBConfig,
    BGE_VERTICAL_BASE_CONFIG,
    setup_db
)

from langchain_huggingface import HuggingFaceEmbeddings




if __name__ == "__main__":
    
    # db = setup_db(model)
    
    
    
    # db = setup_db_openai(multi_thread=False)
    db_config = DBConfig(**BGE_VERTICAL_BASE_CONFIG)
    chat_config = ChatConfig(**GEMINI_FAST_CONFIG)
    text2sql_config = Text2SQLConfig(**TEXT2SQL_MEDIUM_OPENAI_CONFIG)
    prompt_config = PromptConfig(**VERTICAL_PROMPT_BASE)
    
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs = {'device': 'cuda'})
    db_config.embedding = embedding_model
    
    db = setup_db(db_config)
    text2sql = Text2SQL(config = text2sql_config, prompt_config=prompt_config, db = db, max_steps=2)
    
    
    print('Test similarity search company', db.vector_db_company.similarity_search('vinamilk', 2))
    print('Test account')
    print(db.search_return_df('total assets', 2, 'bank'))
    
    prompt = "Compare the Return on Assets (ROA) and Return on Equity (ROE) of Vinamilk and Masan Group for the fiscal year 2023.  Additionally, provide the total assets and total equity for both companies for the same period."
    his, err, tab = text2sql.solve(prompt)
    
    for t in tab:
        print(t)