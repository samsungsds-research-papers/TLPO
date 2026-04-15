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
import glob
import json
import os
import datetime
import shutil
import sys

from dataclasses import asdict
from typing import Union

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from dataset.confusion_detector import get_all_response_consistency, ConsistencyResult, LC_DETECTOR_VERSION, LINGUA_THRESHOLD


def try_parse_json(value: str) -> Union[str, dict, None]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings."
            )
        return value


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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--harness_output_dir", type=str, default="tools/evaluation/harness_output", help="lm-evaluation-harness result dir"
    )
    parser.add_argument(
        "--ignore_english", type=str, default="true", help="ignore_english e.g. 'true', 'false'.. defualt true"
    )
    parser.add_argument(
        "--target_language", type=str, default="ko", help="One of ko, zh, ar, or ja"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="tools/evaluation/tlpo_output",
        type=str,
        metavar="DIR|DIR/file.json",
        help="Path where result metrics will be saved. Can be either a directory or a .json file. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    return parser


def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()


def get_result_harness(k: str, v: str):
    skip_consistency = False
    ndigits = 4
    
    if "gsm8k" in k:
        result = {
                "exact_match": round(v["exact_match,flexible-extract"] * 100, ndigits)
            }
        
        if "gsm8k_platinum_cot_zeroshot" == k:
            skip_consistency = True
    elif "mmmlu" in k:
        result = {
                "acc": round(v["acc,none"] * 100, ndigits)
            }
    elif "bbh_cot_zeroshot" in k:
        result = {
                "exact_match": round(v["exact_match,flexible-extract"] * 100, ndigits)
            }
        skip_consistency = True
    elif "mif" in k:
        result = {
                # "inst_level_loose_acc": round(v["inst_level_loose_acc,none"] * 100, ndigits),
                # "inst_level_strict_acc": round(v["inst_level_strict_acc,none"] * 100, ndigits),
                # "prompt_level_loose_acc": round(v["prompt_level_loose_acc,none"] * 100, ndigits),
                "prompt_level_strict_acc": round(v["prompt_level_strict_acc,none"] * 100, ndigits),
            }
        if "mif_en" == k:
            skip_consistency = True
    elif "non_greedy_robustness" in k:
        result = {
                "acc": round(v["non_greedy_accuracy,none"] * 100, ndigits)
            }
        skip_consistency = True
    elif "gpqa" in k:
        result = {
                "exact_match": round(v["exact_match,flexible-extract"] * 100, ndigits)
            }
        skip_consistency = True
    elif "arc" in k:
        result = {
                "exact_match": round(v["exact_match,remove_whitespace"] * 100, ndigits)
            }
        skip_consistency = True
    else:
        # lcb
        result = {}

    return result, skip_consistency


def get_response_language(json_data: dict) -> str:
    lang_idx = json_data["doc"]["instruction_id_list"].index("language:response_language")
    response_language = json_data["doc"]["kwargs"][lang_idx]["language"]

    if response_language == "zh-cn":
        response_language = "zh"

    return response_language


def run_eval(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        parser = setup_parser()
        args = parse_eval_args(parser)

    if args != None:
        print('*'*50)
        for key, value in vars(args).items():
            print(f" {key}: {value}")
        print('*'*50)

    # ex. '20250624_175313'
    job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ignore_english = True if args.ignore_english.lower()=="true" else False
    target_language = args.target_language.lower()

    output_dir = args.output_dir if args.output_dir else "./"
    if os.path.exists(output_dir) and os.path.isdir(output_dir)==False:
        os.remove(output_dir)    # remove if output_dir is a file

    os.makedirs(output_dir, exist_ok = True)

    harness_result_file = glob.glob(args.harness_output_dir + f"/*.json")
    result_harness = None

    try:
        harness_result_file = harness_result_file[0]
        with open(harness_result_file, "r") as f:
            result_harness = json.load(f)
    except Exception as e:
        print("[error] harness result file open error!")
        return

    if not isinstance(result_harness, dict):
        print("[error] harness result file type error!")
        return

    total_result = {}
    total_result["version"] = LC_DETECTOR_VERSION
    total_result["target_language"] = target_language
    total_result["ignore_english"] = ignore_english
    total_result["lingua_threshold"] = LINGUA_THRESHOLD
    
    result_id = os.path.basename(harness_result_file).split("_")[-1].replace(".json", "")

    for task_name, v in result_harness["results"].items():
        result, skip_consistency = get_result_harness(task_name, v)
        target_task_sample_file_list = []

        # response sample file copy
        total_sample_file_list = glob.glob(args.harness_output_dir + f"/samples_{task_name}_*.jsonl")
        for sample_file_path in total_sample_file_list:
            if result_id in sample_file_path:
                if task_name == "ifeval":
                    if "ifeval_ko" in sample_file_path:
                        continue

                target_task_sample_file_list.append(sample_file_path)
                shutil.copy(sample_file_path, output_dir)

        if skip_consistency:
            consistency = asdict(ConsistencyResult())
        else:
            sample_list = []

            for sample_file_path in target_task_sample_file_list:
                with open(sample_file_path, "r") as f:
                    for line in f:
                        json_data = json.loads(line)

                        if "gsm8k" in task_name:
                            if json_data["filter"] == "flexible-extract":
                                sample_list.append(json_data)
                        elif "mif" in task_name:
                            if "mif_ko" == task_name and json_data["doc_id"] == 67: # label error
                                continue
                            
                            if "language:response_language" in json_data["doc"]["instruction_id_list"]:
                                response_language = get_response_language(json_data)

                                if response_language != target_language:
                                    continue
                            
                            sample_list.append(json_data)
                        else:
                            sample_list.append(json_data)

            response_list = [sample_list[i]["resps"][0][0] for i in range(len(sample_list))]
            response_consistency = get_all_response_consistency(response_list, target_language, ignore_english)
            consistency = asdict(response_consistency)

        result["consistency"] = consistency
        total_result[task_name] = result

    save_path = os.path.join(output_dir, f"./TLPO_eval_{job_id}.json")

    with open(save_path, "w") as f:
        json.dump(total_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_eval()
