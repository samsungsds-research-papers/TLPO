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
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

from dataset.confusion_detector import *


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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset/Bactrian-X-filtered/data", help="dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="train_data", help="result directory"
    )
    parser.add_argument(
        "--target_language", type=str, default="ko", help="One of ko, zh, ar, or ja"
    )
    parser.add_argument(
        "--ignore_english", type=str, default="true", help="ignore_english e.g. 'true', 'false'.. defualt true"
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
    output_dir = args.output_dir

    model_list = [
        "llama",
        "qwen",
        "gemma4b",
        "ministral",
    ]
    target_language = args.target_language
    ie = "iet" if args.ignore_english == "true" else "ief"

    for model_name in model_list:
        data_file = f"{dataset_dir}/{model_name}/{target_language}.jsonl"

        if not os.path.exists(data_file):
            print(f"file not exist - {model_name}")
            continue

        data_list = load_dataset("json", data_files=data_file, split="train")

        dpo_train_data_list = []

        ignore_english = False if "ief" in ie else True

        for d in tqdm(data_list):
            answer_list = d["answer"]
            
            none_confusion_idx_list = []
            confusion_idx_list = []
            
            for i, a in enumerate(answer_list):
                if get_confusion_point(a, target_language, ignore_english) == -1:
                    none_confusion_idx_list.append(i)
                else:
                    confusion_idx_list.append(i)

            if len(none_confusion_idx_list) == 0 or len(confusion_idx_list) == 0:
                pass
            else:
                chosen = answer_list[random.sample(none_confusion_idx_list, 1)[0]]
                rejected = answer_list[random.sample(confusion_idx_list, 1)[0]]
                
                example = {
                    "prompt": d["question"],
                    "chosen": chosen,
                    "rejected": rejected,
                }
            
                dpo_train_data_list.append(example)

        print(f"[{model_name}]")
        print(f"dpo: {len(dpo_train_data_list)}")
        print("------------------------------------------")

        os.makedirs(output_dir, exist_ok = True)
        output_path = os.path.join(output_dir, f"batcrian_dpo_random_{ie}_{model_name}_{target_language}.json")
        write_file(output_path, dpo_train_data_list)


if __name__ == "__main__":
    main()
