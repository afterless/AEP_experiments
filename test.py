import json

def count_jsonl_items(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass
    return count

# Replace 'your_file.jsonl' with the path to your JSONL file
file_path = './evals/persona/openness.jsonl'
num_items = count_jsonl_items(file_path)
print(f"The JSONL file contains {num_items} items.")


