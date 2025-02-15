
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# from vllm import LLM,SamplingParams
import json
import random
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from vllm import LLM,SamplingParams
from datasets import load_dataset
import copy

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        # print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data, ensure_ascii=False, indent=4))


def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        for line in f1:
            data = json.loads(line)
            con.append(data)
    #print(con[0])        
    return con


model_name = "/QwQ-32B-Preview"

cut_id = 1
math = load_file(f'/split_part_0{cut_id}.json')
# random.shuffle(math)
math = math[:10000]



SYSTEM = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection,
backtracing, and iteration to develop well-considered thinking process.
Please structure your response into two main sections: Thought and Solution.
In the Thought section, detail your reasoning process using the specified format:
“‘
<|begin_of_thought|>
{thought with steps separated with "\n\n"}
<|end_of_thought|>
”’
Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.
In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:
“‘
<|begin_of_solution|>
{final formatted, precise, and clear solution}
<|end_of_solution|>
”’
Now, try to solve the following question through the above guidelines:
"""
llm = LLM(model=model_name, tensor_parallel_size=4)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=4096)
messages = []
for i in tqdm(range(0, len(math))):
    question = math[i]['label_conversations'][0]['value']
    information = [{"role":"system", "content":SYSTEM}, {"role":"user", "content": question}]
    message = tokenizer.apply_chat_template(information, tokenize=False, add_generation_prompt=True)
    messages.append(message)
outputs = llm.generate(messages, sampling_params)



file_path = f'/mavis_4k_geo_cut_id{cut_id}_test.json'
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'w', encoding='utf-8') as f1:
    i = -1
    for solution in outputs:        
        i += 1
        data_item = copy.deepcopy(math[i])
        data_item['conversations'] = data_item['label_conversations']
        data_item['conversations'][1]['value'] = solution.outputs[0].text
        f1.write(json.dumps(data_item, ensure_ascii=False) + '\n')





