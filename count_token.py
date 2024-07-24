import json
import numpy as np
import pandas as pd

# Load JSON file
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/worse_data_combined/all_selected_output_iteration_0+1_second_worse.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to count tokens
def count_tokens(text):
    return len(text.split())

# Concatenate the instruction, input, and output for each object and count tokens
token_counts = []
for item in data:
    # combined_text = f"{item['instruction']} {item['input']} {item['output']}"
    combined_text = f"{item['output']}"
    token_count = count_tokens(combined_text)
    # print(f"Token count: {token_count}")
    token_counts.append(token_count)

# Calculate quantiles
quantiles = np.percentile(token_counts, [25, 50, 75, 100])

# Create a DataFrame to display the results
df = pd.DataFrame({
    'Token Counts': token_counts
})

# Print quantiles
quantile_results = {
    '0.25 Quantile': quantiles[0],
    '0.50 Quantile': quantiles[1],
    '0.75 Quantile': quantiles[2],
    'Max Value': quantiles[3]
}

print(quantile_results)
