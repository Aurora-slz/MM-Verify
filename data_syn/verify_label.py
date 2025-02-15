
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# from vllm import LLM,SamplingParams
import json
import random
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM,SamplingParams
from tqdm import tqdm
import re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import base64

def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        for line in f1:
            data = json.loads(line)
            con.append(data)
    print(con[0])        
    return con

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        # print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data, ensure_ascii=False, indent=4))


model_name = "gpt-4o or mm-verify(stage1)"
llm = LLM(
        model=model_name,
    )
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True)
sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=2048,
            stop_token_ids=[],
        )


def inference(tmp_prompt_list):
    inputs = []
    processor = AutoProcessor.from_pretrained("model_name")
    for i in tqdm(range(0, len(tmp_prompt_list))):
        input_prompt = tmp_prompt_list[i]['prompt']
        img_path = tmp_prompt_list[i]['image']
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
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })

    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    res_data = []
    for o in outputs:
        response = o.outputs[0].text
        res_data.append(response)
    return res_data


def create_x(data):
    return f"Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\".\n\nQ: {data['question']}\nA: Let's think step by step.\n{data['solution']}"


def get_label_solution_sampleData(prm_save_path, train_save_path, mm=None):
    
    load_path = prm_save_path
    data = load_file(load_path)
    FROM = 0
    END = len(data)
    data = data[FROM:END]
    print(len(data))
    
    prompt = []
    outputs = []
    all_grade_input = []
    all_grade_output = []
    for i in range(0, len(data)):
        for j in range(0, len(data[i]['orm_data'])):
            # print('='*10)
            question = data[i]['orm_data'][j][0]
            solution = data[i]['orm_data'][j][1]
            # for j in range(0, len(data[i]['prm_data'])):
            #     solution += f"{data[i]['prm_data'][j]['step']}"
            # expected_answer = data[i]['conversations'][-1]['value']
            expected_answer = data[i]['answer']
            image = os.path.join('/data_train/code/sft_intern/slz/math_mm/Data/MathVerse/images', data[i]['image'])
            # prompt_i = f"You are a math teacher. Grade the Solution, verifying correctness step by step. Use Expected Answer to find any erroneous step in the Solution. At the end of the Solution verification, when you give your final grade, write it in the form \"Verification: Is the answer correct (Yes/No)? X\", where X is either Yes or No.\n{question}\nSolution:\n{solution}\nExpected Answer:\n{expected_answer}"
            prompt_i = f"You are a math teacher involves thoroughly Verifying the Solution through a systematic long thinking process before providing the final precise and accurate judgement. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Grade the Solution, verifying correctness step by step. Use Expected Answer to find any erroneous step in the Solution. At the end of the Solution verification, when you give your final grade, write it in the form \"Verification: Is the answer correct (Yes/No)? X\", where X is either Yes or No.\n{question}\nSolution:\n{solution}\nExpected Answer:\n{expected_answer}"
            all_grade_input.append({'prompt':prompt_i, 'image':image})

    all_grade_output = inference(all_grade_input)
    
    data_index = 0
    for i in range(0, len(all_grade_output)):
        response = all_grade_output[i]
        data[i // 20]['orm_data'][i % 20].append({'grade_input': all_grade_input[i]})
        data[i // 20]['orm_data'][i % 20].append({'grade_output': response})
        # if(i + 1 % 20 == 0):
        #     data_index += 1
        
    save_file(data, train_save_path+f"_SubData_{FROM}_{END}")
    # save_file(data, '/cpfs/29f69eb5e2e60f26/code/sft_intern/lh/slz/ReST-MCTS/outputs/sft_data/train_prm_sftData_qwen2.5-7b-instruct_test.json')






    



if __name__ == '__main__':
    
    prm_save_path = '/solution_sample.json'
    train_save_path = '/solution_sample_verifyData.json'
    get_label_solution_sampleData(prm_save_path, train_save_path)