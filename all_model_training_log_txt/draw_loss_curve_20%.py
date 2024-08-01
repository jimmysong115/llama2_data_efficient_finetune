# import json
# import matplotlib.pyplot as plt

# # 读取 JSON 文件
# with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/classiferi_only_20%_log.json', 'r') as f:
#     data = json.load(f)

# # 提取 step, loss 信息
# steps = list(range(1, len(data) + 1))
# losses = [item['loss'] for item in data]

# # 绘制曲线
# plt.figure(figsize=(10, 6))
# plt.plot(steps, losses, label='Training Loss', color='blue')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.title('Training Loss vs Step')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/.test.png')

import json
import matplotlib.pyplot as plt
import os

# 假设所有的 JSON 文件都存储在一个文件夹中
json_folder = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt'

# 获取文件夹中的所有 JSON 文件
# json_files = [f for f in os.listdir(json_folder) if f.endswith('.json') and f !='selected_full_data_log.json' and f!='selected_10%_data_log.json' and f!='selected_15%_data_log.json' and f!='selected_20%_data_1024_log.json']

json_files = ['/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_data_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/classifier_only_20%_log.json',
'/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/diversity_only_20%_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/non_iteration_bert_20%_log.json',
'/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/complete_random_20%_log.json']

# json_files = ['/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/complete_random_20%_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_data_log.json']
# json_files = ['/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/classifier_only_20%_log.json','/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_data_log.json']

# json_files = ['/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/selected_20%_data_log.json']

# plt.figure(figsize=(12, 8))

plt.figure(figsize=(12, 8))

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
    plt.plot(steps, losses, label=label,linewidth=2)

plt.xlabel('Step',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.title('Training Loss vs Step for Multiple Models',fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True,linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
plt.savefig('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/20%_compare_1024.png')

# import json
# import matplotlib.pyplot as plt
# import os

# # 假设所有的 JSON 文件都存储在一个文件夹中
# json_folder = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt'

# # 获取文件夹中的所有 JSON 文件
# json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# plt.figure(figsize=(12, 8))

# # 遍历每个 JSON 文件
# for json_file in json_files:
#     with open(os.path.join(json_folder, json_file), 'r') as f:
#         data = json.load(f)
        
#     # 提取 step 和 loss 信息
#     steps = list(range(1, len(data) + 1))
#     losses = [item['loss'] for item in data]
    
#     # 绘制曲线
#     plt.plot(steps, losses, label=json_file.split('.')[0], linewidth=2)

# plt.xlabel('Step', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.title('Training Loss vs Step for Multiple Models', fontsize=16)
# plt.legend(loc='upper right', fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.7)

# # 设置 x 和 y 轴的刻度字体大小
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.show()

# plt.savefig('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_model_training_log_txt/test_all_1.png')


