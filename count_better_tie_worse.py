import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_instructions(iter1_data, better_iter2, tie_iter2, worse_iter2, better_key, tie_key, worse_key):
    count = {"better": 0, "tie": 0, "worse": 0}
    for entry in iter1_data:
        instruction = entry["instruction"]
        if any(instruction == item["instruction"] for item in better_iter2):
            count[better_key] += 1
        elif any(instruction == item["instruction"] for item in tie_iter2):
            count[tie_key] += 1
        elif any(instruction == item["instruction"] for item in worse_iter2):
            count[worse_key] += 1
    return count

# Load the JSON files
# better_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_better_iter1.json')
# worse_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_worse_iter1.json')
# tie_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_tie_iter1.json')

# better_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_better_iter2.json')
# worse_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_worse_iter2.json')
# tie_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_tie_iter2.json')

# better_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_non_iteration_WizardLM_better_iter1.json')
# worse_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_non_iteration_WizardLM_worse_iter1.json')
# tie_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_non_iteration_WizardLM_tie_iter1.json')

# better_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_non_iteration_WizardLM_better_iter2.json')
# worse_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_non_iteration_WizardLM_worse_iter2.json')
# tie_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_non_iteration_WizardLM_tie_iter2.json')

better_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_better_iter1.json')
worse_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_worse_iter1.json')
tie_iter1 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_tie_iter1.json')

better_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_better_iter2.json')
worse_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_worse_iter2.json')
tie_iter2 = load_json('/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_tie_iter2.json')

# Count instructions
better_count = count_instructions(better_iter1, better_iter2, tie_iter2, worse_iter2, 'better', 'better', 'tie')
tie_count = count_instructions(tie_iter1, better_iter2, tie_iter2, worse_iter2, 'better', 'tie', 'worse')
worse_count = count_instructions(worse_iter1, better_iter2, tie_iter2, worse_iter2, 'tie', 'worse', 'worse')

# Aggregate counts
total_counts = {
    "better": better_count["better"] + tie_count["better"],
    "tie": better_count["tie"] + tie_count["tie"] + worse_count["tie"],
    "worse": tie_count["worse"] + worse_count["worse"]
}

# Print the results
print("20%_full_data_Vicuna_comparison:")
print("Better count:", total_counts["better"])
print("Tie count:", total_counts["tie"])
print("Worse count:", total_counts["worse"])
