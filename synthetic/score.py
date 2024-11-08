import argparse
import json

# Set up the argument parser
parser = argparse.ArgumentParser(description="Parse JSON file from command-line arguments.")
parser.add_argument("file_path", type=str, help="Path to the JSON file.", default="gpt-4o-generated-v2-gpt-4o-mini.json")


args = parser.parse_args()

with open(args.file_path, 'r') as f:
    data = json.load(f)
    
die_query = 0
scores = 0
input_tokens = 0

output_tokens = 0
ids = set()

bug_ids = dict()

for qa in data:
    if qa['id'] not in bug_ids:
        bug_ids[qa['id']] =[]
    bug_ids[qa['id']].append(qa)
    if qa['id'] in ids:
        print(f"Duplicate id: {qa['id']}")
        continue
    ids.add(qa['id'])
    response = qa['response'].strip()
    if response == "":
        die_query += 1
    if response.endswith('----:|\n'):
        die_query += 1
    scores += qa['evaluate']
    input_tokens += qa['input_token']
    output_tokens += qa['output_token']
# for id, qa in bug_ids.items():
    
#         score = 0
#         selected_qa = qa[0]
#         for q in qa:
#             if q['evaluate'] > score:
#                 selected_qa = q
#                 score = q['evaluate']
                
#         response = selected_qa['response'].strip()
#         if response == "":
#             die_query += 1
#         if response.endswith('----:|\n'):
#             die_query += 1
#         scores += selected_qa['evaluate']
#         input_tokens += selected_qa['input_token']
#         output_tokens += selected_qa['output_token']
#         ids.add(selected_qa['id'])
    
scores = scores / 103

print(f"Results for {args.file_path}")
print(f"Number of unique queries: {len(ids)}")
print(f"Number of queries: {len(data)}")
print(f"Average score: {scores}")
print(f"Number of die queries: {die_query}, ({die_query/103*100}%)")
print(f"Average input tokens: {input_tokens / 103}")
print(f"Average output tokens: {output_tokens / 103}")