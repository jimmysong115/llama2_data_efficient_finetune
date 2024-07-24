import transformers,torch
from transformers import LlamaForCausalLM, AutoTokenizer
 
#下载好的hf模型地址
hf_model_path = '/llm_gluster_new/model/Llama-2-7b-hf'
# hf_model_path = '/llm_gluster_new/model/Llama-2-7b-chat-hf'
 
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=hf_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
 
sequences = pipeline(
    # 'Season nine of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 9, 2005. The ninth season concluded after 14 episodes on December 7, 2005. All of the episodes in the ninth season were written and directed by Trey Parker.\n\n Question: Does this imply that "The character Kenny from South Park has never died."? Yes, no, or maybe?.',
    # 'what is the capital of China?',
    # 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    "What is Pitiquito, Mexico?",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=128,
    # temperature = 0.7,
)

print(sequences)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")





