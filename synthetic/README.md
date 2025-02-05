# Generate synthetic data


### Merge the data 
Run the following scripts to merge the questions

```bash
python generate.py --task merge_questions
```

### Generate synthetic question

```bash
python generate.py --task generate_questions --multi_thread True
```

### Generate MCQ 

```bash
llm=gpt-4o-mini
task=generate_mcq
multi_thread=True
path=../data/sql_v0.jsonl

python generate.py --llm $llm --task $task --multi_thread $multi_thread --path $path
```