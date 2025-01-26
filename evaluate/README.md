# Evaluation kit

## Synthetic evaluation



Normally, the question file is named `generated_questions.jsonl` and the solution file is named `{model_name}__{question_version}.jsonl`

### Arguments:
- llm: LLM for SQL Generation and/or Validation
- multi_thread: Run the task in multi-thread
- using_cache: For generation task `generation.py`, using cache help speed up the code generation process from `Text2SQL` object
- version: Get the question version in `../data/generated_question.jsonl`
- task: get the task

And many more ...

### For generating solution 
#### Pre request: You must have a question file in `data` folder. 

Generate the SQL code for the task

Run the following script (remember change the argument accordingly)

```bash
llm=gpt-4o-mini 
multi_thread=True 
version=v0

python generate.py --llm $llm --version $version --multi_thread $multi_thread
```


### For evaluating the difficulty
#### Pre request: You must have a solution file in `data` folder. 
Score the probability that the question can be answered with given data. (weak label)

Run the following script (remember change the argument accordingly)

```bash
llm=gpt-4o
multi_thread=True 
task=qa_quality

python validate.py --llm $llm --task $task --multi_thread $multi_thread
```

Prefer using strong LLM for better valuation