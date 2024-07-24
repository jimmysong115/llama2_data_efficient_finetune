import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
 
# Original version
# model_path = "LinkSoul/Chinese-Llama-2-7b"
# 4 bit version
model_path = "/gluster_pdc_llm/user/songjielin/initial_checkpoint/Llama-2-7b-hf"
 
 
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto'
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
 
instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""
 
while True:
    text = input("请输入 prompt\n")
    if text == "q":
        break
    prompt = instruction.format(text)
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)


# first_line = '5    8.5'
# print(first_line.split())
# score1, score2 = map(float, first_line.split())
# print(score1)
# print(score2)