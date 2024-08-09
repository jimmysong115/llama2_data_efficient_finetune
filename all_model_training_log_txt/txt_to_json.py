import json

# Read the text file and convert it into a list of dictionaries
input_file_path = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/classifier_only_20%_1024_log.txt'

with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Convert each line to a dictionarys
data = [eval(line.strip()) for line in lines]

# Convert list of dictionaries to JSON format
json_data = json.dumps(data, indent=4)

# Save the JSON data to a file
output_file_path = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/classifier_only_20%_1024_log.json'
with open(output_file_path, 'w') as json_file:
    json_file.write(json_data)

output_file_path
