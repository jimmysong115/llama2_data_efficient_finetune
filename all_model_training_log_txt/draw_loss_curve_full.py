

import json
import matplotlib.pyplot as plt
import os

# 假设所有的 JSON 文件都存储在一个文件夹中
json_folder = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt'

# 获取文件夹中的所有 JSON 文件
# json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# json_files = ['/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/complete_random_20%_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_data_log.json']
json_files = ['/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_full_data_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_data_log.json'
,'/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_15%_data_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_10%_data_log.json']

# plt.figure(figsize=(12, 8))

plt.figure(figsize=(14, 10))

# 遍历每个 JSON 文件
for json_file in json_files:
    with open(os.path.join(json_folder, json_file), 'r') as f:
        data = json.load(f)
        
    # 提取 step 和 loss 信息
    # steps = list(range(1, len(data) + 1))

    steps = [i*2 for i in range(len(data))]
    losses = [item['loss'] for item in data]

    # 提取文件名作为标签
    label = os.path.basename(json_file).replace('_log.json', '')
    
    # 绘制曲线
    plt.plot(steps, losses, label=label)

plt.xlabel('Step',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Training Loss vs Step for Multiple Models',fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plt.savefig('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_vs_full_2steps.png')

