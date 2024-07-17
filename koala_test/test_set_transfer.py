import json

# Define the input and output file paths
input_file_path = '/gluster_pdc_llm/user/songjielin/koala_test/koala_test_set.jsonl'
output_file_path = '/gluster_pdc_llm/user/songjielin/koala_test/koala_test_set.json'

# Read the jsonl file and convert it to the desired json format
data_list = []

with open(input_file_path, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        new_data = {
            "instruction": data["prompt"],
            "input": ""
        }
        data_list.append(new_data)

# Save the new json format to a file
with open(output_file_path, 'w') as file:
    json.dump(data_list, file, indent=2)

