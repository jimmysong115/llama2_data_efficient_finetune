import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data (你已有的数据)
data = {
    'Training Data Size': [5, 10, 15, 20, 60,'full'],
    'koala': [(66, 35, 79), (69, 43, 68), (74, 26, 80), (75, 38, 67), (78, 33, 69),(1,1,1)],
    'LIMA': [(117, 45, 138), (155, 39, 105), (159, 47, 94), (145, 52, 103), (152,46, 102),(1,1,1)],
    'self_instruct': [(88,53, 111), (83, 60, 109), (102, 53, 97), (107, 64, 81), (106, 55, 91),(1,1,1)],
    'WizardLM': [(75, 40, 103), (60, 50, 108), (76, 43, 99), (81, 44, 93), (76, 42, 100),(1,1,1)],
    'Vicuna': [(28, 12, 40), (42, 11, 27), (44, 13, 23), (41, 17, 22), (43, 10, 27),(1,1,1)]
}

def calculate_wining_score(results):
    win, tie, lose = results
    # return (win - lose) / (win + lose + tie) + 1
    result = round((win - lose) / (win + lose + tie) + 1, 2)
    print(result)
    return result

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# 计算每个点的总和
total_curve = []
for i in range(len(data['Training Data Size'])):
    total_win = sum(data[curve][i][0] for curve in ['koala', 'LIMA', 'self_instruct', 'WizardLM', 'Vicuna'])
    total_tie = sum(data[curve][i][1] for curve in ['koala', 'LIMA', 'self_instruct', 'WizardLM', 'Vicuna'])
    total_lose = sum(data[curve][i][2] for curve in ['koala', 'LIMA', 'self_instruct', 'WizardLM', 'Vicuna'])
    total_curve.append((total_win, total_tie, total_lose))

# 使用 calculate_wining_score 函数计算总曲线的 Wining Score
df['Total'] = [calculate_wining_score(result) for result in total_curve]

# 绘制图表
plt.figure(figsize=(8, 4))
x_positions = range(len(df['Training Data Size']))

# 先绘制已有的曲线
# for curve in ['koala', 'LIMA', 'self_instruct', 'WizardLM', 'Vicuna']:
#     plt.plot(x_positions, df[curve], marker='o', label=curve)

# 再绘制总曲线
plt.plot(x_positions, df['Total'],  marker='o', linestyle='--', color='orange', label='Total')

plt.xlabel('Training Data Size (%)')
plt.ylabel('Total Wining Score')
plt.title('Total Wining Score vs Training Data Size')
plt.xticks(x_positions, ['5%', '10%', '15%', '20%', '60%','full'])

# 突出显示 y=1.0 的线
plt.axhline(y=1.0, color='black', linewidth=1)

# 移除框架线条
plt.grid(axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().yaxis.grid(True)
plt.gca().xaxis.grid(False)

plt.legend()

# 设置纵坐标刻度格式为保留两位小数
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.savefig('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_plot_gpt4_evaluate_curve/Wining_Score_vs_Training_Data_Size_with_Total.png')
