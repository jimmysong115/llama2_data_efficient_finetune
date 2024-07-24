# # 导入 vLLM 所需的库
 
# from vllm import LLM, SamplingParams
 
# # 定义输入提示的列表，这些提示会被用来生成文本
 
# prompts = [
 
# # "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] hello",
 
# "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] 意大利国的总统是",
 
# # "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] 尼日利亚的首都是",
 
# # "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST] 人工智能的未来是",
 
# ]
 
# # 定义采样参数，temperature 控制生成文本的多样性，top_p 控制核心采样的概率
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512,presence_penalty=1.1)
 
 
# # 初始化 vLLM 的离线推理引擎，这里选择的是 "/root/chinese-llama2" 模型
# llm = LLM(model="NousResearch/Llama-2-7b-hf")
 
# # 使用 llm.generate 方法生成输出文本。
# # 这会将输入提示加入 vLLM 引擎的等待队列，并执行引擎以高效地生成输出
# outputs = llm.generate(prompts, sampling_params)
 
# # 打印生成的文本输出
# for output in outputs:
#     prompt = output.prompt # 获取原始的输入提示
#     generated_text = output.outputs[0].text # 从输出对象中获取生成的文本
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# from http import HTTPStatus
# from openai import OpenAI


# def multi_round():


#     base_url = "http://localhost:8000/v1"
#     client = OpenAI(api_key="EMPTY", base_url=base_url)

#     messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
#                 {'role': 'user', 'content': 'what is the capital of China'
#                 }]
    

#     response = client.chat.completions.create(
#         # model="Qwen1.5-7B-Chat",
#         model="NousResearch/Llama-2-7b-hf",
#         # model="Qwen1.5-32B-Chat-AWQ",
#         messages=messages,
#         stream=False,
#         max_tokens=2048,
#         temperature=0.9,
#         presence_penalty=1.1,
#         top_p=0.8)
    
#     if response:
#         content = response.choices[0].message.content
#         print(content)
            

#     else:
#         print("Error:", response.status_code)


# if __name__ == '__main__':
#     multi_round()


from openai import OpenAI
import json



# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8002/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# test_json_file = '/gluster_pdc_llm/user/songjielin/Vicuna/Vicuna_test_set.json'
# test_json_file = '/gluster_pdc_llm/user/songjielin/LIMA/LIMA_test_set.json'
# test_json_file = '/gluster_pdc_llm/user/songjielin/WizardLM/WizardLM_testset.json'
# test_json_file = '/gluster_pdc_llm/user/songjielin/koala_test/koala_test_set.json'
test_json_file = '/gluster_pdc_llm/user/songjielin/self_instruct/self_instruct_testset.json'

# test_json_file = '/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_seleted_input_data.json'

# 从文件中读取JSON数据
with open(test_json_file, 'r') as file:
    data = json.load(file)
    count = 0
    for entry in data:

        # if count >= 5:
        #     break
        
        instruction = entry["instruction"]
        input_text = entry["input"]
        prompt = f"{instruction}.\n{input_text}"

        # print(len(prompt))

        if len(prompt)>13000:
            prompt = prompt[:13000]

        print(prompt)

        completion = client.completions.create(
            model="llama2_7b_hf_classfier_only_20%",
            prompt=prompt,
            temperature=0.7,
            max_tokens=400,
            presence_penalty=1.1)
            # temperature=0.6,
            # top_p=0.95)

        count += 1
        
        if completion:
            content = completion.choices[0].text
            print("result: " + completion.choices[0].text)
            entry["output"] = content
            # break
        else:
            print("Error!")

# 将结果写回文件
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/seletced_classifier_only_data/20%_data_decode/self_instruct_test_decode.json', 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

# with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_seleted_output_decode.json', 'w') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)