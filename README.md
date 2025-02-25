# Chatbot_financial_statement

#### Update 23-02-2025: EVF is a bank

#### Update 22-02-2025: New Text2SQL UI + add new db

#### Update 17-02-2025: Just need the evaluation dataset

## File structures

### Agent

This is the codebase for Text2SQL Agent and Chatbot

- Agent: Text2SQL solver, Text2SQL configs
- Chatbot: Normal Chatbot, Chatbot with Semantic Layers.
- Prompts
- Implementation of MCTS

### ETL

This is the codebase for database setup and DB utils for Text2SQL solver and Chatbot (access to Postgre and vectordb) 

- DBManager: Connect to RDB and doing vector search and rerank. Also include Semantic Layers
- Connector: Utils and setup for RDB and vectordb
- Ratio_index, const, etl: Create financial ratio, merge financial statements

### Synthetic

- Generate synthetic data, including SQL and MCQs-related questions.

### Evaluate

- Generate SQL query based on the given question
- Evaluate the quality of the SQL generated

### Page

- Host the Chatbot via steamlit

### Trainer

- GPRO Trainer

## LLM Available

### Gemini:

- Gemini 2.0 Flash
- Gemini 2.0 Flash Thinking Exp

### GPT

- GPT 4o mini (Local test: 0.65)
- GPT 4o

### Deepseek

- Deepseek-chat

### Qwen

- Qwen2.5-Coder-3B-SFT (Local test: 0.72)
- Qwen2.5-Coder-3B-KTO
- Qwen2.5-Coder-3B-DPO
- Qwen2.5-Coder-1.5B-SFT
- Qwen2.5-Coder-1.5B-KTO
- Qwen2.5-Coder-1.5B-DPO

## Database design

- Horizontal: Each account/ratio is a columns in the main table
- Vertical: Each account/ratio is a row in the main table


## Prompting Strategy
- General: 2-step Text2sql. First asking LLM to analyze the problem and choose which category do they want to access. Then adding snapshot of the table into prompt, so it can correctly select the right column.
- Reasoning: After having snapshot, ask LLM to generate SQL directly to solve the problem
- Include debugging

## Setup guide
<details>
<summary>Ignore</summary>
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
</details>

### Setup maunally



Clone the reporitoty and create environment

```bash
git clone https://github.com/hung20gg/chatbot_financial_statement.git
cd chatbot_financial_statement
git clonehttps://github.com/hung20gg/llm.git

conda create -y -n text2sql
conda activate text2sql
pip install -r requirements.txt
```

**Prepare env file**
Create a `.env` file and put all the necessary key into it
```
OPENAI_API_KEY=your_api_key
GEMINI_API_KEY=your_api_key

DB_NAME=
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=

MONGO_DB_HOST=
MONGO_DB_USER=
MONGO_DB_PASSWORD=
MONGO_DB_PORT=

EMBEDDING_SERVER_URL=http://localhost:8080
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

RERANKER_SERVER_URL=http://localhost:8081
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

LOCAL_EMBEDDING=True


DEEPSEEK_HOST=https://api.deepseek.com
DEEPSEEK_API_KEY=your_api_key

LLM_HOST=http://localhost:8000/v1
```
Then run
```
# Linux, macos
source .env

# Window
.env
```

#### Local Embedding
**Note** Build the TEI local and run the following scripts (check the [TEI repo](https://github.com/huggingface/text-embeddings-inference) for setup)

- For embedding:
```bash
model=BAAI/bge-base-en-v1.5
text-embeddings-router --model-id $model --port 8080
```

- For Reranker (it is ok not to have reranker)
```bash
model=BAAI/bge-reranker-v2-m3
text-embeddings-router --model-id $model --port 8081
```

#### Setup database
Using any existing database or using Docker Image for:
- Postgre
- MongoDB (optional)

Create database via this scripts (notice the version)

```bash
python setup.py --preprocess v3 --force True --local True --vectordb chromadb
```

For using data of 200 companies
```bash
python setup.py --preprocess v3.2 --force True --local True --vectordb chromadb
```

**Note:** If you are not using local embedding, remove `--local True` and replace with `--openai True`

Run the `test.py` file to check the setup status
```bash
python test.py
```

### DB In the pipeline
- ChromaDB/ Milvus (Storing the embedding)
- PostgreSQL (Storing the data)
- MongoDB (Storing the user message)



Check and add the index for full-text search in [ETL\index_full_text_search.md](ETL\index_full_text_search.md)

## Train & evaluate

### Database

By dividing into different database, we can test on the scalability of the model when handling changes in database

- Training DB (v3): 100 companies
- Evaluating DB (v3.2): 200+ companies + QoQ ratio + 4 nearest quarter ratio

### Training strategy
Using [LLaMA-Factory CLI](https://github.com/hiyouga/LLaMA-Factory) for LoRA tuning (r=64)
- Model choice: Qwen2.5 Coder 1.5B/3B (7B if funded)
- SFT (good)
- Proof Preference Learning (DPO, KTO) is not good  (done)
- SFT with external dataset + merge-kit.

### Evaluation set

- Real question from reports.
- Synthetic data as eval set.

### MCQ evaluation

- v1: 1 SQL → 1 MCQ
- v2: 1 SQL → multiple MCQs. Average them, or get int(avg(scores))

## Versioning v1 (training) and v2 (testing) database

Notice: Do not change the content in `csv` folder

Changes of training and testing dataset:

- Increase in the number of companies
- Increase in the number of accounts and query format
- Changes in industry average

Change the `DB_NAME` in `.env` file to another version to load

## TO DO
- [x] Add new v3.2 companies into the database
- [x] Text2SQL UI
- [ ] Evaluation dataset
- [ ] Moving logging & vectordb to Elastic Search 
- [ ] FastAPI backend 
- [ ] Write unit tests  