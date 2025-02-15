import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import logging

import re

from openai import AzureOpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from prompts.ext_ans import demo_prompt
import json
from vllm import LLM, SamplingParams
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

model_path = "/Llama-3.2-3B-Instruct"
# model = 'a'
model = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.7)
tokenizer = model.get_tokenizer()
sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])

def save_json(data, path):
    with open(path, 'w') as f:
        data_json = json.dumps(data, indent=4)
        f.write(data_json)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        for line in f1:
            data = json.loads(line)
            con.append(data)
    print(con[0])        
    return con


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\nModel response: {response}"
    full_prompt = f"{demo_prompt}\n\n\n{test_prompt}\nExtracted answer: "
    return full_prompt


def extract_answer(model, response, problem, quick_extract=False, batch_vllm=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # quick extraction
    if quick_extract:
        logging.info("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass
    
    if(batch_vllm):
        return "vllm"
    
    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        
        outputs = model.generate(full_prompt, sampling_params)
        res_data = []
        for j in range(0, len(outputs)):
            output = outputs[j]
            prompt = output.prompt
            response = output.outputs[0].text
            res_data.append(response)
        # extraction = model.get_response(user_prompt=full_prompt)
        return res_data[0]
    except Exception as e:
        logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logging.info(e)

    return ""



def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='/math_cot_Llama-3.2-11B-Vision-Instruct_ocrCaption.json')
    parser.add_argument('--extract_file_path', type=str, default='response')
    parser.add_argument('--extractSaved_file_path', type=str, default='response')
    parser.add_argument('--extractDiscard_file_path', type=str, default='response')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The max number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    parser.add_argument('--key', type=str, default='testmini')
    
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))

    args = parser.parse_args()
    return args

import re
def extract_answer_gptVerify(content):
    pattern_yes = r'Is the answer correct(?: \(Yes\/No\))?\? Yes'
    if re.search(pattern_yes, content) is not None:
        return 'Yes'
    else:
        return 'No'

def main():
    logging.info("MathVista: Extract Answers - Start")
    args = parse_args()

    # args
    label = args.response_label

    logging.info(f"Reading {args.results_file_path}...")
    # results = read_json(args.results_file_path)
    # results = load_file_2(args.results_file_path)
    results = read_json(args.results_file_path)
    END = 1000
    results = results[:END]

    tmp = {}
    turn1_saved_data = []
    turn1_discard_data = []
    image_length = []
    for i in tqdm(range(0, len(results))):
        positive = 0
        negtive = 0
        prompt_list = []
        query = results[i]['conversations'][0]['value']
        real_answer_context = results[i]['conversations'][1]['value']        

        for j in range(0, len(results[i]['orm_data'])):
            gpt_yn_context = results[i]['orm_data'][j][-1]['grade_output']
            mcts_answer_context = results[i]['orm_data'][j][1]
            
            mcts_answer_context_prompt = create_test_prompt(demo_prompt, query, mcts_answer_context)
            mcts_tmp = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': mcts_answer_context_prompt}],
                tokenize=False,
            )
            real_answer_context_prompt = create_test_prompt(demo_prompt, query, real_answer_context)
            real_tmp = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': real_answer_context_prompt}],
                tokenize=False,
            )
            
            prompt_list.append(mcts_tmp)
            prompt_list.append(real_tmp)

            outputs = model.generate(prompt_list, sampling_params, use_tqdm=False)
            mcts_answer = outputs[0].outputs[0].text
            real_answer = outputs[1].outputs[0].text
            gpt_yn = extract_answer_gptVerify(gpt_yn_context)
            tmp_dict = {}
            tmp_dict['answerExtract_mcts'] = mcts_answer.replace('<|start_header_id|>assistant<|end_header_id|>', '').strip()
            tmp_dict['answerExtract_real'] = real_answer.replace('<|start_header_id|>assistant<|end_header_id|>', '').strip()
            tmp_dict['gptExtract_yn'] = gpt_yn
            results[i]['orm_data'][j].append(tmp_dict)
            
            mcts_answer = tmp_dict['answerExtract_mcts']
            real_answer = tmp_dict['answerExtract_real']
            gpt_yn = tmp_dict['gptExtract_yn']

            # if(mcts_answer == real_answer and gpt_yn == 'Yes'):
            #     turn1_saved_data.append(results[i])
            # elif(mcts_answer != real_answer and gpt_yn == 'No'):
            #     turn1_saved_data.append(results[i])
            # else:
            #     turn1_discard_data.append(results[i])
            if(mcts_answer == real_answer and gpt_yn == 'Yes'):
                positive += 1
            elif(mcts_answer != real_answer and gpt_yn == 'No'):
                positive += 1
            else:
                negtive += 1
            image_length.append(min(positive, negtive))

    
    save_json(results, args.extract_file_path)
    save_json(turn1_saved_data, args.extractSaved_file_path)
    print(len(turn1_saved_data))
    save_json(turn1_discard_data, args.extractDiscard_file_path)
    print(len(turn1_discard_data))
    
    logging.info(f"Saved results to {args.extract_file_path}")
    logging.info("MathVista: Extract Answers - Finish")



if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
