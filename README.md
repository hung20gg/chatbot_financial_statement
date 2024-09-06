# group_2_chatbot_financial_statement

## LLM setup

Clone my separate library for LLM 

For sample folder 

```
cd group_2_chatbot_financial_statement/sample 
git clone https://github.com/hung20gg/llm.git
```

Create a `.env` file in folder `sample` and add `OPENAI_API_KEY = sk-...`

## Database setup (run only the first time)

First, you need to create the environment

```
python -m venv env
```

Install the library

```
cd group_2_chatbot_financial_statement && pip install -r requirements.txt
```

To setup a database, simply pull a Docker image via

```
docker run bitnami/postgresql \
		-e POSTGRES_PASSWORD=12345678 \
		-e POSTGRES_DB=test_db \
		-p 5432:5433 \
		-d postgres
```

run docker in another port `5433` to avoid conflict with local port

Then run the python scripts to connect and add data to the database

```
cd group_2_chatbot_financial_statement/sample && python setup_db.py
```

You can test the chat ability in `chatbot.ipynb`