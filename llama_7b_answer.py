# # 导入 vLLM 所需的库
 
# from vllm import LLM, SamplingParams
 
# # 定义输入提示的列表，这些提示会被用来生成文本
 
# prompts = [
 
# "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] hello",
 
# # "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] 意大利国的总统是",
 
# # "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] 尼日利亚的首都是",
 
# # "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] 人工智能的未来是",
 
# ]
 
# # 定义采样参数，temperature 控制生成文本的多样性，top_p 控制核心采样的概率
# sampling_params = SamplingParams(temperature=0.9, top_p=0.8, max_tokens=512,presence_penalty=1.1)
 
 
# # 初始化 vLLM 的离线推理引擎，这里选择的是 "/root/chinese-llama2" 模型
# llm = LLM(model="/gluster_pdc_llm/user/songjielin/initial_checkpoint/Llama-2-7b-hf")
 
# # 使用 llm.generate 方法生成输出文本。
# # 这会将输入提示加入 vLLM 引擎的等待队列，并执行引擎以高效地生成输出
# outputs = llm.generate(prompts, sampling_params)
 
# # 打印生成的文本输出
# for output in outputs:
#     prompt = output.prompt # 获取原始的输入提示
#     generated_text = output.outputs[0].text # 从输出对象中获取生成的文本
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-hf"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
    
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output)