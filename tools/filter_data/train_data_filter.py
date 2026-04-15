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
import gzip
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from dataset.confusion_detector import get_confusion_point


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


def check_filter(question):
    question_lower = question.lower()
    for filter in filter_list:
        if filter.lower() in question_lower:
            return False
    return True


def write_file(path: str, data_list: list):
    with open(path, "a", encoding="UTF-8") as f:
        for d in data_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset", help="dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/Bactrian-X-filtered/data", help="result directory"
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


def main():
    parser = setup_parser()
    args = parse_eval_args(parser)

    if args != None:
        print('*'*50)
        for key, value in vars(args).items():
            print(f" {key}: {value}")
        print('*'*50)

    dataset_dir = args.dataset_dir
    target_language = args.target_language
    # https://huggingface.co/datasets/MBZUAI/Bactrian-X
    dataset_path = os.path.join(dataset_dir, f"data/{target_language}.json.gz")

    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        lang_data = json.load(f)

    len(lang_data)

    list_filter = []

    for i, example in enumerate(lang_data):
        if example["input"] is not None and len(example["input"])>=1:
            question = f'{example["instruction"]}\n{example["input"]}'
        else:
            question = f'{example["instruction"]}'

        if get_confusion_point(question, target_language) == -1 and check_filter(question):
            list_filter.append({"question": question})

    len(list_filter)


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok = True)

    out_path = f"{output_dir}/{target_language}.json"
    with open(out_path, "wt", encoding="utf-8") as f:
        json.dump(list_filter, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()