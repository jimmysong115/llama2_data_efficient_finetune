# import requests
# import json
# import re

# # 定义API调用函数
# def call_gpt_api(messages):
#     url = "http://gbu.jp02-a30-apisix-online.baidu-int.com/gbu/rest/v1/ai_chat/openai_service"
    

#     headers = {
#         "apikey": "synclub-gvnerwtgwe24nme",
#     }

#     # #hono
#     # params = {
#     #     "model_name": "deploy_gpt4_32k",
#     #     # "model" : "gpt-3.5-turbo",
#     #     "context": messages,
#     #     "temperature": "1",
#     #     "max_token": 8000,
#     #     "ans_token": 4000,
#     #     "stream": False,
#     #     "top_p": "1"
#     # }

#     #else
#     params = {
#         "model_name": "deploy_gpt4_0613",
#         # "model" : "gpt-3.5-turbo",
#         "context": messages,
#         "temperature": "1",
#         "max_tokens": 4000,
#         "ans_token": 1000,
#         "stream": False,
#         "top_p": "1"
#     }

    

#     response = requests.post(url, headers=headers, json=params)

#     if response.status_code == 200:
#         return response.json(),response.status_code
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)
#         return None,response.status_code


# input_file = "/ssd2/songjielin/gpt_teacher/original_data/single_round_distillation_0419.jsonl"
# output_file_no_change = "/ssd2/songjielin/gpt_teacher/single_round_distillation_0419_no_change_0628.jsonl"
# output_file_add = "/ssd2/songjielin/gpt_teacher/single_round_distillation_0419_add_0628.jsonl"
# fail_txt = "/ssd2/songjielin/gpt_teacher/single_round_distillation_0419_fail_0628.txt"



# with open(input_file, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# output_data_no_change = []
# output_data_add = []

# for line in lines[:]:
#     data = json.loads(line)
#     conversations = data.get("conversations", [])

#     messages = []
#     messages.append({
#     "text": "You are a helpful and precise assistant for checking the quality of the answer.",
#     'role_type': 'system'
#     })
#     model_response = ""


#     dialogue = ""
#     system_message = None
#     count = 0



# [Question]
# Question
# [The Start of Assistant 1’s Answer]
# Answer 1
# [The End of Assistant 1’s Answer]
# [The Start of Assistant 2’s Answer]
# Answer 2
# [The End of Assistant 2’s Answer]
# We would like to request your feedback on the performance of two AI assistants in response to the
# user question displayed above.
# Please rate the helpfulness, relevance, accuracy,
# level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where
# a higher score indicates better overall performance.
# Please first output a single line containing only two
# values indicating the scores for Assistant 1 and
# 2, respectively. The two scores are separated by
# a space. In the subsequent line, please provide
# a comprehensive explanation of your evaluation,
# avoiding any potential bias and ensuring that the
# order in which the responses were presented does
# not affect your judgment.

#     messages.append({"text": dialogue, "role_type": "user"})


    
#     # print(messages)

#     # 调用GPT API生成数据
#     api_response,status_code = call_gpt_api(messages)

#     if api_response:
#         assistant_message = api_response["data"]["content"]
#         print(assistant_message)


        
#     else:
        
#         if status_code == 429:
#             break
#         else:
#             print(messages)
#             with open(fail_txt, "a") as file:  # 使用追加模式打开文件
#                 file.write(str(messages) + "\n")  # 将消息写入文件并换行
#             output_data_no_change.append(data)
        


# # # 将生成的数据写回JSONL文件
# # with open(output_file_add, 'w', encoding='utf-8') as f:
# #     for item in output_data_add:
# #         f.write(json.dumps(item, ensure_ascii=False) + '\n')

# # with open(output_file_no_change, 'w', encoding='utf-8') as f:
# #     for item in output_data_no_change:
# #         f.write(json.dumps(item, ensure_ascii=False) + '\n')

# print("数据生成完成并保存至output.jsonl")


import requests
import json

def call_gpt_api(messages):
    url = "http://gbu.jp02-a30-apisix-online.baidu-int.com/gbu/rest/v1/ai_chat/openai_service"
    headers = {
        "apikey": "synclub-gvnerwtgwe24nme",
    }

    params = {
        "model_name": "deploy_gpt4_32k",
        "context": messages,
        "temperature": "1",
        "max_tokens": 30000,
        "ans_token": 2000,
        "stream": False,
        "top_p": "1"
    }

    response = requests.post(url, headers=headers, json=params)

    if response.status_code == 200:
        return response.json(), response.status_code
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, response.status_code

# input_file_1 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_20%_data/Vicuna_test_decode.json"
# input_file_2 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_full_data/Vicuna_test_decode.json"
# output_file_higher_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_better_iter1.json"
# output_file_tie_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_tie_iter1.json"
# output_file_lower_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_worse_iter1.json"

# output_file_no_change = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_full_data_Vicuna_no_change_iter1.json"


# input_file_1 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_output_no_change_again.json"
# input_file_2 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_shuffled.json"
# output_file_higher_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_shuffled_worse.json"
# output_file_lower_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_shuffled_better.json"

# output_file_no_change = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_diversity_1600_data/all_output_no_change_again2.json"

# input_file_2 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_20%_data/koala_test_decode.json"
# input_file_1 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_complete_random_data/20%_data_decode/koala_test_decode.json"
# output_file_higher_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_random_koala_worse_iter2.json"
# output_file_tie_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_random_koala_tie_iter2.json"
# output_file_lower_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_random_koala_better_iter2.json"

# output_file_no_change = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_random_koala_no_change_iter2.json"

input_file_1 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_20%_data/Vicuna_test_decode.json"
input_file_2 = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_classifier_only_data/20%_data_decode/Vicuna_test_decode.json"
output_file_higher_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_better_iter1.json"
output_file_tie_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_tie_iter1.json"
output_file_lower_score = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_worse_iter1.json"

output_file_no_change = "/gluster_pdc_llm/user/songjielin/extracted_120k_data/20%_selected_vs_classifier_only_Vicuna_no_change_iter1.json"

with open(input_file_1, 'r', encoding='utf-8') as f1, open(input_file_2, 'r', encoding='utf-8') as f2:
    data_1 = json.load(f1)
    data_2 = json.load(f2)

output_data_no_change = []
output_data_add = []
output_higher_score = []
output_lower_score = []
output_tie_score = []

count = 0

for entry1, entry2 in zip(data_1, data_2):


    # if count>=5:
    #     break
    count += 1

    
    question = f"{entry1['instruction']}\n{entry1['input']}"
    answer1 = entry1["output"]
    answer2 = entry2["output"]

    dialogue = f"""
    [Question]
    {question}
    [The Start of Assistant 1’s Answer]
    {answer1}
    [The End of Assistant 1’s Answer]
    [The Start of Assistant 2’s Answer]
    {answer2}
    [The End of Assistant 2’s Answer]
    We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
    """

    messages = [{"text": "You are a helpful and precise assistant for checking the quality of the answer.", "role_type": "system"},
                {"text": dialogue, "role_type": "user"}]

    api_response, status_code = call_gpt_api(messages)

    if api_response:
        assistant_message = api_response["data"]["content"]
        print(assistant_message)

        # 获取评分并比较
        first_line = assistant_message.split('\n')[0]
        try:
            # score1, score2 = map(int, first_line.split())

            score1, score2 = map(float, first_line.split())
            # 将浮点数转换为整数
            int_score1 = int(score1)
            int_score2 = int(score2)
            if score1 > score2:
                output_higher_score.append(entry1)
            elif score1 == score2:
                output_tie_score.append(entry1)
            else:
                output_lower_score.append(entry1)
        except ValueError as e:
            print(f"Error parsing scores: {e}")
            output_data_no_change.append(entry1)
    else:
        if status_code == 429:
            break
        else:
            print(messages)
            # with open(fail_txt, "a") as file:
            #     file.write(str(messages) + "\n")
            output_data_no_change.append(entry1)

# 将结果保存到相应的文件
with open(output_file_higher_score, 'w', encoding='utf-8') as f:
    json.dump(output_higher_score, f, ensure_ascii=False, indent=4)

with open(output_file_tie_score, 'w', encoding='utf-8') as f:
    json.dump(output_tie_score, f, ensure_ascii=False, indent=4)

with open(output_file_lower_score, 'w', encoding='utf-8') as f:
    json.dump(output_lower_score, f, ensure_ascii=False, indent=4)

with open(output_file_no_change, 'w', encoding='utf-8') as f:
    json.dump(output_data_no_change, f, ensure_ascii=False, indent=4)

print("数据生成完成并保存至output.json")
