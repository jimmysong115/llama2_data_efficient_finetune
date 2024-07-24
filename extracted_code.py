
# alpaca && evol_instruct && unnatural && self_instruct
import json
import random
def sample_jsonl(input_file, output_jsonl_file, output_json_file, sample_size):
    # Read all lines from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if sample_size is greater than the number of lines in the file
    if sample_size > len(lines):
        print(f"Sample size {sample_size} is greater than the number of available lines {len(lines)}. Using all lines instead.")
        sample_size = len(lines)
    
    # Randomly sample the lines
    sampled_lines = random.sample(lines, sample_size)
    
    # Write the sampled lines to the output JSONL file
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    # Parse the sampled lines into a list of JSON objects
    sampled_json_objects = [json.loads(line) for line in sampled_lines]
    
    # Write the list of JSON objects to the output JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_json_objects, f, ensure_ascii=False, indent=4)

# Usage example
input_file = '/gluster_pdc_llm/user/songjielin/self_instruct/all_instances_82K.jsonl'
output_jsonl_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/self_instruct_sampled_output.jsonl'
output_json_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/self_instruct_sampled_output.json'
sample_size = 15000

sample_jsonl(input_file, output_jsonl_file, output_json_file, sample_size)


## LaMini
# import json
# import random

# def sample_jsonl_with_filter(input_file, output_jsonl_file, output_json_file, sample_size):
#     # Read all lines from the input JSONL file
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     # Filter lines based on the specified conditions
#     valid_lines = [
#         line for line in lines if 'instruction_source' in json.loads(line) and
#         json.loads(line)['instruction_source'] in ['generated_flan', 'generated_p3', 'original_flan', 'original_p3']
#     ]
    
#     # Check if sample_size is greater than the number of valid lines
#     if sample_size > len(valid_lines):
#         print(f"Sample size {sample_size} is greater than the number of available valid lines {len(valid_lines)}. Using all valid lines instead.")
#         sample_size = len(valid_lines)
    
#     # Randomly sample the valid lines
#     sampled_lines = random.sample(valid_lines, sample_size)
    
#     # Write the sampled lines to the output JSONL file
#     with open(output_jsonl_file, 'w', encoding='utf-8') as f:
#         f.writelines(sampled_lines)
    
#     # Parse the sampled lines into a list of JSON objects
#     sampled_json_objects = [json.loads(line) for line in sampled_lines]
    
#     # Write the list of JSON objects to the output JSON file
#     with open(output_json_file, 'w', encoding='utf-8') as f:
#         json.dump(sampled_json_objects, f, ensure_ascii=False, indent=4)

# # Usage example
# input_file = '/gluster_pdc_llm/user/songjielin/LaMini-instruction/data/merged.jsonl'
# output_jsonl_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/LaMini_sampled_output.jsonl'
# output_json_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/LaMini_sampled_output.json'
# sample_size = 15000

# sample_jsonl_with_filter(input_file, output_jsonl_file, output_json_file, sample_size)


# # Dolly && dynosaur && Longform
# import json
# import random
# from collections import defaultdict

# def sample_jsonl_with_balanced_category(input_file, output_jsonl_file, output_json_file, total_sample_size):
#     # Read all lines from the input JSONL file
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     # Group the lines by category
#     category_groups = defaultdict(list)
#     for line in lines:
#         data = json.loads(line)
#         category = data.get('source', None)
#         if category is not None:
#             category_groups[category].append(line)
    
#     # Determine the number of categories and the sample size per category
#     num_categories = len(category_groups)
#     if num_categories == 0:
#         raise ValueError("No categories found in the data.")
    
#     sample_size_per_category = total_sample_size // num_categories
    
#     # Sample from each category
#     sampled_lines = []
#     for category, lines in category_groups.items():
#         if len(lines) < sample_size_per_category:
#             print(f"Category '{category}' has less than {sample_size_per_category} entries, taking all {len(lines)} entries.")
#             sampled_lines.extend(lines)
#         else:
#             sampled_lines.extend(random.sample(lines, sample_size_per_category))
    
#     # Write the sampled lines to the output JSONL file
#     with open(output_jsonl_file, 'w', encoding='utf-8') as f:
#         f.writelines(sampled_lines)
    
#     # Parse the sampled lines into a list of JSON objects
#     sampled_json_objects = [json.loads(line) for line in sampled_lines]
    
#     # Write the list of JSON objects to the output JSON file
#     with open(output_json_file, 'w', encoding='utf-8') as f:
#         json.dump(sampled_json_objects, f, ensure_ascii=False, indent=4)

# # Usage example
# input_file = '/gluster_pdc_llm/user/songjielin/LongForm/dataset/train.jsonl'
# output_jsonl_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/LongForm_sampled_output.jsonl'
# output_json_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/LongForm_sampled_output.json'
# total_sample_size = 15000

# sample_jsonl_with_balanced_category(input_file, output_jsonl_file, output_json_file, total_sample_size)


