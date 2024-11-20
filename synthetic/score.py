import argparse
import json
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))

# Set up the argument parser
parser = argparse.ArgumentParser(description="Parse JSON file from command-line arguments.")
parser.add_argument("file_path", type=str, help="Path to the JSON file.", default="gpt-4o-generated-v2-gpt-4o-mini.json")


args = parser.parse_args()

with open(args.file_path, 'r') as f:
    data = json.load(f)
    
with open(os.path.join(current_dir, 'gpt-4o-generated-v2-scored.json'), 'r') as f:
    qa_4o = json.load(f)
    
selected_ids = set()
for qa in qa_4o:
    selected_ids.add(qa['id'])
    
die_query = 0
scores = 0
input_tokens = 0

output_tokens = 0
time = 0
ids = set()

accepted_qa = 0

for qa in data:
    if qa['id'] in selected_ids:
        accepted_qa += 1
        ids.add(qa['id'])
        response = qa['response'].strip()
        if response == "":
            die_query += 1
        if response.endswith('----:|\n'):
            die_query += 1
        scores += qa['evaluate']
        input_tokens += qa['input_token']
        output_tokens += qa['output_token']
    # time += qa['time']

    
total = accepted_qa
scores = scores / total

print(f"Results for {args.file_path}")
print(f"Number of unique queries: {len(ids)}")
print(f"Number of queries: {total}")
print(f"Average score: {scores}")
print(f"Number of die queries: {die_query}, ({die_query/total*100}%)")
print(f"Average input tokens: {input_tokens / total}")
print(f"Average output tokens: {output_tokens / total}")
print(f"Average time: {time / total}")
print(f"Average input tokens: {data[-1]['input_token'] / total}")
print(f"Average output tokens: {data[-1]['output_token'] / total}")