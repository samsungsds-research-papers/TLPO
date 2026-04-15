"""
The MIT License

Copyright (c) 2026 Samsung SDS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import os
import json
from vllm import LLM, SamplingParams

from datasets import load_dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


language_names = [
    "English", "Mandarin", "Chinese", "Spanish", "Hindi", 
    "French", "Arabic", "Bengali", "Russian", "Portuguese", 
    "Urdu", "Indonesian", "German", "Japanese", "Swahili", 
    "Marathi", "Telugu", "Turkish", "Korean", "Vietnamese", 
    "Tamil", "Punjabi", "Javanese", "Persian", "Italian", "Thai",       

    "영어", "만다린어", "중국어", "스페인어", "힌디어", 
    "프랑스어", "아랍어", "벵골어", "러시아어", "포르투갈어", 
    "우르두어", "인도네시아어", "독일어", "일본어", "스와힐리어", 
    "마라티어", "텔루구어", "터키어", "한국어", "베트남어", 
    "타밀어", "펀자브어", "자바어", "페르시아어", "이탈리아어", "태국어",               
]


exception_words = [
    'translate', '번역',
    'python', '파이썬',
    'html',
    'java', '자바',
    'xml',
    'script','스크립트',
    'SQL',
    'css',
    'Hello World',
    'Hello, World',
    'def ',
    'C++', 'C#',
    'http', 'https',
    'git', 'github', 'gitlab',
    'bash',
]


filter_list = language_names + exception_words


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset/Bactrian-X-filtered/data", help="dataset directory"
    )
    parser.add_argument(
        "--model_type", type=str, default="hf", help="Name of model e.g. `hf` or `llama` or `qwen`"
    )
    parser.add_argument(
        "--model_path", type=str, default="pretrained_model/Meta-Llama-3.1-8B-Instruct/", help="model full path"
    )
    parser.add_argument(
        "--target_language", type=str, default="ko", help="One of ko, zh, ar, or ja"
    )

    return parser


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue


def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()


def get_data_list(path: str) -> list:
    data_list = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def write_file(path: str, data_list: list):
    with open(path, "a", encoding="UTF-8") as f:
        for d in data_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def check_filter(question):
    question_lower = question.lower()
    for filter in filter_list:
        if filter.lower() in question_lower:
            return False
    return True


###############################################################################################
def main():
    parser = setup_parser()
    args = parse_eval_args(parser)

    if args != None:
        print('*'*50)
        for key, value in vars(args).items():
            print(f" {key}: {value}")
        print('*'*50)

    model_type = args.model_type
    model_path = args.model_path
    target_language = args.target_language

    dataset_dir = args.dataset_dir
    data_path = f"{dataset_dir}/{target_language}.json"
    data_list = load_dataset("json", data_files=data_path, split="train")

    llm = LLM(model=model_path)

    if model_type == "qwen":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_type == "llama":
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, n=16, repetition_penalty=1.02)
    else:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, n=16)

    save_path = f"{dataset_dir}/{model_type}/{target_language}.jsonl"

    for i, d in enumerate(data_list):
        try:
            question = d["question"]

            messages = [
                {"role": "user", "content": question}
            ]

            if model_type == "qwen":
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Set to False to strictly disable thinking
                )
                outputs = llm.generate([text], sampling_params)
            else:
                outputs = llm.chat(messages, sampling_params=sampling_params)

            answer_list = [outputs[0].outputs[i].text for i in range(16)]

            temp = {
                "question": question,
                "answer": answer_list,
            }
        
            write_file(save_path, [temp])
        except Exception as e:
            print(f"skip - {i}")


if __name__ == "__main__":
    main()
