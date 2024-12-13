# Chatbot_financial_statement

## Major Update 2

- Streamlit is conflict with similarity search ???
- Done evaluation for ML1
- For fastest inference, use `llm = Gemini('gemini-1.5-flash-8b')` and `sql_llm = Gemini('gemini-1.5-flash-002')`

## Major Update

### Database new design

- Separate 3 type of financial reports: Bank, non-bank and security firms (VA regulation)
- Store both company general information and ownership
- Pre-calculated financial ratio
- Average indicator based on collected stock-code

Each of them will have 2 tables, one store information and the other store mapping code

In total, there are 13 tables:

- bank_financial_report
- non_bank_financial_report
- map_category_code_non_bank
- map_category_code_bank

### Prompting strategies

- General: 2-step Text2sql. First asking LLM to analyze the problem and choose which category do they want to access. Then adding snapshot of the table into prompt, so it can correctly select the right column.
- Reasoning: After having snapshot, ask LLM to generate SQL directly to solve the problem
- Partial sql. Instead of query to find the solution, breakdown steps and solve it one-by-one
- Include debugging

## LLM setup

Clone my separate library for LLM


```
cd group_2_chatbot_financial_statement && git clone https://github.com/hung20gg/llm.git

```

Create a `.env` file in folder `sample` and add `OPENAI_API_KEY = sk-...`

Also add config for database

```
DB_NAME=test_db
DB_USER=postgres
DB_PASSWORD=12345678
DB_HOST=localhost
DB_PORT=5433
```

## Database setup (run only the first time)

First, you need to create the environment

```
cd ../
python -m venv env
```

Install the library

```
pip install -r requirements.txt
```

To setup a database, simply pull a Docker image via

```
docker run bitnami/postgresql \
		-e POSTGRES_PASSWORD=12345678 \
		-e POSTGRES_DB=test_db \
		-p 5432:5433 \
		-d postgres
```

Run docker in another port `5433` to avoid conflict with local port

Then run the python scripts to connect and add data to the database

```
cd ETL && python connector.py
```

Check and add the index for full-text search in [ETL\index_full_text_search.md](ETL\index_full_text_search.md)
