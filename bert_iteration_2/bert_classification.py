import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch
from torch.nn import CrossEntropyLoss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# # 加载 positive 和 negative 数据
# with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/better_data_combined/all_selected_output_iteration_0+1_second_better.json', 'r') as f:
#     positive_data = json.load(f)

# with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/worse_data_combined/all_selected_output_iteration_0+1_second_worse.json', 'r') as f:
#     negative_data = json.load(f)

# 加载 positive 和 negative 数据
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/better_data_combined/all_selected_output_iteration_0+1+2_second_better.json', 'r') as f:
    positive_data = json.load(f)

with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/worse_data_combined/all_selected_output_iteration_0+1+2_second_worse.json', 'r') as f:
    negative_data = json.load(f)

# 添加标签
for item in positive_data:
    item['label'] = 1

for item in negative_data:
    item['label'] = 0

# 合并数据
data = positive_data + negative_data

# 转换为 DataFrame
df = pd.DataFrame(data)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# 计算类别权重
num_negative = train_df['label'].value_counts()[0]
num_positive = train_df['label'].value_counts()[1]
total_samples = num_negative + num_positive

print(f'Number of negative samples: {num_negative}')
print(f'Number of positive samples: {num_positive}')
print(f'Total number of samples: {total_samples}')

weight_negative = float(total_samples / num_negative)
weight_positive = float(total_samples / num_positive)

# class_weights = torch.tensor([weight_negative, weight_positive]).cuda()
class_weights = torch.tensor([1.0,9.0]).cuda()

# 创建 Dataset 对象
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 加载 tokenizer 和 model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained('/gluster_pdc_llm/user/songjielin/extracted_120k_data/bert_iteration_1/results/second_test/checkpoint-160')

# 设置最大长度
max_length = 256

# 编写 tokenization 函数
def tokenize_function(examples):
    # 批量处理时，遍历每个样本
    combined_texts = [
        f"Instruction: {instruction} Input: {input_text} Output: {output_text}"
        for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return tokenizer(combined_texts, padding="max_length", truncation=True, max_length=max_length)

# 应用 tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 删除不需要的列
train_dataset = train_dataset.remove_columns(["instruction", "input", "output"])
test_dataset = test_dataset.remove_columns(["instruction", "input", "output"])

# 设置格式
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results/second_test",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # 每个epoch保存一次
    save_steps=500,  # 每500步保存一次
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to=[],  # 禁用 wandb
    save_total_limit=1  # 最多保留1个checkpoint
)

# 自定义 Trainer
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 初始化 Trainer
trainer = WeightedLossTrainer(
    model=BertForSequenceClassification.from_pretrained('/gluster_pdc_llm/user/songjielin/extracted_120k_data/bert_iteration_1/results/second_test/checkpoint-160', num_labels=2),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,  # 添加 tokenizer 以确保 padding 和 truncation 的一致性
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()

# 评估模型
results = trainer.evaluate()
print("Evaluation results:", results)

# 获取验证集的预测结果
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# 打印混淆矩阵
conf_matrix = confusion_matrix(true_labels, pred_labels)

# 使用 pandas 创建一个 DataFrame，用于美化输出混淆矩阵
df_conf_matrix = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# 打印混淆矩阵
print("Confusion Matrix:")
print(df_conf_matrix)
