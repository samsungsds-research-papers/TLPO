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



import torch
import os
import numpy as np
import pandas as pd
from random import shuffle
import copy
import logging
import pdb
import json
import math
import re

from dataset.dataset_base import BaseDataset

class RegexFilter():
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback


    def _clean_number_string(self, number_string):
        cleaned_string = re.sub(r'[,.]+$', '', number_string.replace(',', ''))
        cleaned_string = cleaned_string.replace('$', '')

        try:
            value = float(cleaned_string)
        except ValueError:
            value = cleaned_string
        
        return value


    def apply(self, resp: list[list[str]]) -> list[list[str]]:
        match = self.regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m]
                if match:
                    match = match[0]
                else:
                    match = self.fallback
            match = self._clean_number_string(match.strip())
        else:
            match = self.fallback

        return match

        

class DatasetGSM8K(BaseDataset):

    def __init__(self,
                 data_params,
                 model_type,
                 fabric,
                 tokenizer,
                 per_device_batch_size):

        
        super().__init__(data_params,
                        model_type,
                        fabric,
                        tokenizer,
                        per_device_batch_size)
            
        self.question_column = 'question'
        self.answer_column = 'answer'

        self.gt_filter = RegexFilter(regex_pattern=r"#### (\-?[0-9\.\,]+)", group_select=0)
        self.res_filter = RegexFilter(regex_pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)


    def _load_examples(self,
                        filepath,
                        do_truncate):


        examples = super()._load_examples(filepath, do_truncate)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'gsm8k_instructions.json')

        with open(json_path, "r") as f:
            instuction_all = json.load(f)

        instruction = instuction_all.get(self.target_language, None)
        if instruction==None:
            raise ValueError(f'invalid target_language: {self.target_language}')
        
            
        for idx in range(len(examples)):
            q = examples[idx]['question']
            examples[idx]['question'] = instruction.format(question=q)

        return examples

        
    def _get_content(self, question: str, index: int):
        return question

    
    def get_score(self, gt_answer, res_answer):
        gt_answer_num = self.gt_filter.apply(gt_answer)
        res_answer_num = self.res_filter.apply(res_answer)

        score = 1.0 if gt_answer_num==res_answer_num else 0.0

        return score
        

        
