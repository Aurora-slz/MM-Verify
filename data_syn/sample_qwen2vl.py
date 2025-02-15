import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import json  
import random
import re
from datasets import load_dataset
from sample_prompt import ACTION
from tqdm import tqdm

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        for line in f1:
            data = json.loads(line)
            con.append(data)
    print(con[0])        
    return con


model_name = "/Qwen2-VL-7B-Instruct"
llm = LLM(
        model=model_name, tensor_parallel_size=1
    )
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True)


sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=4096,
            stop_token_ids=[151645,151643],
        )



class TreeNode:
    def __init__(self, value, path, depth):
        self.value = value  
        self.path = path  
        self.children = [] 
        self.depth = depth
        self.prompt = []
        self.action_path = []
    

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def get_input_prompt(self):
        return ' '.join(self.prompt)




def inference(input_prompt, depth, img_path):

    img_path = os.path.join("<img_path>", img_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": input_prompt},
            ],
        },
    ]
    processor = AutoProcessor.from_pretrained("/Qwen2-VL-7B-Instruct")
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }]

    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    res_data = []
    for o in outputs:
        response = o.outputs[0].text
        res_data.append(response)
    return res_data[0]


def build_tree(question, root_value, depth, branch_factor, img_path, sample_number):
    
    root = TreeNode(root_value, ["root"], 1)
    root.prompt = [question]
    root.action_path = ['x']
    
    
    def add_children(node, current_depth):
        
        if current_depth >= depth+1:
            return
        
        for i in range(0, sample_number):
                 
            child_value = node.value * branch_factor + i 
            child_path = node.path + [f"child{i+1}"]

            input_prompt = ACTION[1].format(question=node.get_input_prompt())
            output = inference(input_prompt, current_depth, img_path)


            child_node = TreeNode(child_value, child_path, current_depth)
            child_node.prompt = node.prompt + [output]
            child_node.action_path = node.action_path + [f'a{1}']
            node.add_child(child_node)

    add_children(root, 1)
    return root

candidate_answer = []

def search_tree_dfs(node):
    if node is None:
        return
    
    if(len(node.children) == 0):
            candidate_answer.append(node.prompt)
    else:
        for child in node.children:
            search_tree_dfs(child)


def load_data():
    
    dataset = load_file("/mm_verify_candidata_data.json")
    return dataset
        


if __name__ == "__main__":
    
    load_path = os.getenv('load_path')
    output_file = os.getenv('output_file')
    SAMPLE_NUMBER = int(os.getenv('SAMPLE_NUMBER'))
    print(load_path)
    print(output_file)
    
    data = load_file(load_path)
    with open(output_file, 'w', encoding='utf-8') as f1:
        for i in tqdm(range(0, len(data))):
            orm_data = []
            img_path = data[i]['image']
            query = data[i]['conversations'][0]['value']
            gold_answer = data[i]['conversations'][-1]['value']
            candidate_question = query
            candidate_answer = []

            root = build_tree(question=candidate_question, root_value=1, depth=1, branch_factor=6, img_path=img_path, sample_number=SAMPLE_NUMBER)
            search_tree_dfs(root)
            
            data[i]['orm_data'] = candidate_answer
            f1.write(json.dumps(data[i], ensure_ascii=False)+'\n')
            f1.flush()
        