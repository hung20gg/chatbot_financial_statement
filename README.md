# Chatbot_financial_statement

## Major Update 3
- Having universal code 
- Ready to deploy
- Setup 2 Docker image, MongoDB and Text2SQL
- If you have gpu, you can also use Text Embedding Inference (TEI) image for fast embedding
- Gonna make an image in the future 

## Prompting Strategy
- General: 2-step Text2sql. First asking LLM to analyze the problem and choose which category do they want to access. Then adding snapshot of the table into prompt, so it can correctly select the right column.
- Reasoning: After having snapshot, ask LLM to generate SQL directly to solve the problem
- Partial sql. Instead of query to find the solution, breakdown steps and solve it one-by-one
- Include debugging

## Setup guide

- Make `run.sh` file executable
```bash
chmod +x run.sh
```

- For CPU (Using OpenAI Embedding)
```bash
./run.sh --openai True
```

- For GPU (Self-hosted Embedding Server)
```bash
./run.sh local-embedding --local True
```
- For GPU, Including Reranker 
```bash
./run.sh local-server --local True 
```

Or you can build the TEI local and run the following scripts

- For embedding:
```bash
model=BAAI/bge-base-en-v1.5
text-embeddings-router --model-id $model --port 8080
```

- For Reranker
```bash
model=BAAI/bge-reranker-v2-m3
text-embeddings-router --model-id $model --port 8081
```

## DB In the pipeline
- ChromaDB (Storing the embedding)
- PostgreSQL (Storing the data)
- MongoDB (Storing the user message)




Check and add the index for full-text search in [ETL\index_full_text_search.md](ETL\index_full_text_search.md)
