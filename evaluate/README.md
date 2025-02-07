# Evaluation kit

## Synthetic evaluation


Normally, the question file is named `generated_questions.jsonl` and the solution file is named `{model_name}__{question_version}.jsonl`

### Common Arguments:
- llm: LLM for SQL Generation and/or Validation
- multi_thread: Run the task in multi-thread
- using_cache: For generation task `generation.py`, using cache help speed up the code generation process from `Text2SQL` object
- version: Get the question version in `../data/generated_question.jsonl` or self-defined version (get all question and named with that version)
- task: Get the task
- path: Path to the question
- max_workers: Max worker for multi-thread
- mcq_path: Path to ter Multiple-choice question

And many more ...

## Warning: If you are not sure what you are doing, DONOT use multi-thread!!!

### For generating solution 
#### Pre request: You must have a question file in `data` folder. 

Generate the SQL code for the task

Run the following script (remember change the argument accordingly)

```bash
llm=gpt-4o-mini
multi_thread=True 
version=v3
path=../data/generated_questions.json

python generate.py --llm $llm --version $version --multi_thread $multi_thread --path $path
```

Using *correction* for refine SQL

```bash
llm=gemini-2.0-flash-thinking-exp-01-21
version=v1
path=../data/generated_questions.json
template=openai

python generate.py --llm $llm --version $version --path $path --enhance correction
```

Generate SQL code for evaluation dataset `sql_v0.jsonl`
```bash
llm=gpt-4o-mini
multi_thread=True 
version=your_version_here
path=../data/sql_v0.jsonl

python generate.py --llm $llm --version $version --multi_thread $multi_thread --path $path
```


### For re-create messages (for training)
#### Pre request: You must have a solution file in `data` folder.

Generate the message for the original code


```bash
llm=deepseek-chat
multi_thread=True 
version=v4
task=generate_messages
path=../data/deepseek-chat__v0_good.jsonl

python generate.py --llm $llm --version $version --multi_thread $multi_thread --task $task --path $path
```


### For evaluating the difficulty
#### Pre request: You must have a solution file in `data` folder.
Score the probability that the question can be answered with given data. (weak label)

Run the following script (remember change the argument accordingly)

```bash
llm=gpt-4o-mini
multi_thread=True 
task=qa_quality
path=../data/gpt-4o-mini__v3.jsonl

python validate.py --llm $llm --task $task --multi_thread $multi_thread --path $path
```

Prefer using strong LLM for better valuation

### For evaluating the generation with MCQ
#### Pre request: You must have a solution file and MCQ file in `data` folder. 

Answer the MCQ question corresponding to the SQL question

```bash
llm=gpt-4o-mini
multi_thread=True 
task=evaluate
path=../data/gpt-4o-mini__v3.jsonl
mcq_path=../data/mcq_v0.jsonl

python validate.py --llm $llm --task $task --multi_thread $multi_thread --path $path
```

**Note:** The scoring function is current have some bug. Use the scoring under cell *Grade Cell* in `check.ipynb`