import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import json
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 加载训练好的BERT模型
model_name = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/bert_iteration_3/results/second_test /checkpoint-120'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 自定义 compute_metrics 函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': (preds == labels).mean()
    }

# 加载大的JSON文件
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_15%_data/all_initial_data_15%.json', 'r') as f:
    large_data = json.load(f)

# 创建 Dataset 对象
large_dataset = Dataset.from_list(large_data)

# 编写 tokenization 函数
def tokenize_function(examples):
    combined_texts = [
        f"Instruction: {instruction} Input: {input_text} Output: {output_text}"
        for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return tokenizer(combined_texts, padding="max_length", truncation=True, max_length=256)

# 应用 tokenization
large_dataset = large_dataset.map(tokenize_function, batched=True)

# 删除不需要的列
large_dataset = large_dataset.remove_columns(["instruction", "input", "output"])
large_dataset.set_format("torch")

# 使用 Trainer 进行预测
training_args = TrainingArguments(
    per_device_eval_batch_size=8,
    output_dir="./test",
    report_to=[],  # 禁用 wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

# 获取预测结果
predictions = trainer.predict(large_dataset)
neg_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 0].numpy()

# 加载小的JSON文件
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/worse_data_combined/all_selected_output_iteration_0+1+2+3_second_worse.json', 'r') as f:
    small_data = json.load(f)

# 初始化句子编码模型
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# 将小JSON文件中的数据编码为向量
small_texts = [
    f"Instruction: {item['instruction']} Input: {item['input']} Output: {item['output']}"
    for item in small_data
]
small_embeddings = sbert_model.encode(small_texts)

# 定义计算相似度的函数
def calculate_similarity(large_item, small_embeddings):
    large_text = f"Instruction: {large_item['instruction']} Input: {large_item['input']} Output: {large_item['output']}"
    large_embedding = sbert_model.encode([large_text])[0]
    similarities = cosine_similarity([large_embedding], small_embeddings)
    max_similarity = similarities.max()
    return max_similarity

# 计算加权得分
weighted_scores = []
weight_neg_prob = 0.7  # 设置negative概率的权重
weight_similarity = 0.3  # 设置相似度的权重

for i, item in enumerate(large_data):
    similarity = calculate_similarity(item, small_embeddings)
    weighted_score = weight_neg_prob * neg_probs[i] + weight_similarity * similarity
    weighted_scores.append((item, weighted_score))

# 根据加权得分进行排序
weighted_scores.sort(key=lambda x: x[1], reverse=True)

for item, score in weighted_scores[:15492]:
    print(score)


# 设置筛选阈值，可以选择前N个或者根据具体的分数筛选
top_n = 15492  # 选择前15492个
filtered_data = [item for item, score in weighted_scores[:top_n]]

# 保存筛选结果
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_15%_data/all_selected_data_15%_without_worse.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)
