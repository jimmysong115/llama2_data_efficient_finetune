import pandas as pd
import matplotlib.pyplot as plt

# Sample data for the five curves
# data = {
#     'Training Data Size': [5, 10, 15, 20, 60],
#     'koala': [(66, 35, 79), (69, 43, 68), (74, 26, 80), (75, 38, 67), (78, 33, 69)],
#     'LIMA': [(155, 39, 105), (117, 45, 138), (159, 47, 94), (145, 52, 103), (152,46, 102)],
#     'self_instruct': [(88,53, 111), (83, 60, 109), (102, 53, 97), (107, 64, 81), (106, 55, 91)],
#     'WizardLM': [(75, 40, 103), (60, 50, 108), (76, 43, 99), (81, 44, 93), (76, 42, 100)],
#     'Vicuna': [(42, 11, 27), (28, 12, 40), (44, 13, 23), (41, 17, 22), (43, 10, 27)]
# }

data = {
    'Training Data Size': [5, 10, 15, 20, 60,'full'],
    'koala': [(66, 35, 79), (69, 43, 68), (74, 26, 80), (75, 38, 67), (78, 33, 69),(1,1,1)],
    'LIMA': [(117, 45, 138), (155, 39, 105), (159, 47, 94), (145, 52, 103), (152,46, 102),(1,1,1)],
    'self_instruct': [(88,53, 111), (83, 60, 109), (102, 53, 97), (107, 64, 81), (106, 55, 91),(1,1,1)],
    'WizardLM': [(75, 40, 103), (60, 50, 108), (76, 43, 99), (81, 44, 93), (76, 42, 100),(1,1,1)],
    'Vicuna': [(28, 12, 40), (42, 11, 27), (44, 13, 23), (41, 17, 22), (43, 10, 27),(1,1,1)]
}

# Calculate the Wining Score for each curve
def calculate_wining_score(results):
    win, tie, lose = results
    result = (win - lose) / (win + lose + tie) + 1
    # print(result)
    return result

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Calculate the Wining Scores
for curve in ['koala', 'LIMA', 'self_instruct', 'WizardLM', 'Vicuna']:
    df[curve] = df[curve].apply(calculate_wining_score)

# Plot the data
plt.figure(figsize=(8, 4))
# Use a fixed number of points for the x-axis positions
x_positions = range(len(df['Training Data Size']))

for curve in ['koala', 'LIMA', 'self_instruct', 'WizardLM', 'Vicuna']:
    plt.plot(x_positions, df[curve], marker='o', label=curve)

plt.xlabel('Training Data Size (%)')
plt.ylabel('Wining Score')
plt.title('Wining Score vs Training Data Size')
plt.xticks(x_positions, ['5%', '10%', '15%', '20%', '60%','full'])


# Highlight the y=1.0 line
plt.axhline(y=1.0, color='black', linewidth=1)

# Remove the frame and vertical grid lines
plt.grid(axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().yaxis.grid(True)
plt.gca().xaxis.grid(False)
# plt.tick_params(axis='x', which='both', length=0)


plt.legend()
# plt.grid(True)
# plt.show()
plt.savefig('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_plot_gpt4_evaluate_curve/Wining_Score_vs_Training_Data_Size_test.png')