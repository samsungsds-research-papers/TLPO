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
import pandas as pd
import numpy as np
from random import shuffle
import copy
import logging
import tqdm  
import pdb
import json
import math
import re
import random
import pathlib
import gzip  

from dataset.dataset_base import BaseDataset


class DatasetBactrianX(BaseDataset):

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

        self.question_column_1 = 'instruction'
        self.question_column_2 = 'input'
        self.answer_column = 'output'


    def _load_examples(self,
                        filepath,
                        do_truncate):

        filepath_target = filepath.format(lang=self.target_language)
        examples_target = self._load_bactrianx_examples(filepath_target, do_truncate)        
        self.logger.info("[rank {}] load {}, len={}".format(self.fabric.global_rank, filepath_target, len(examples_target)))

        
        # size filter, exclude too long questions
        examples_target = [example for example in examples_target if len(example['question'])<=500]
        
        return examples_target


    def _load_bactrianx_examples(self,
                            filename,
                            do_truncate):

        per_device_batch_size=self.per_device_batch_size
        global_rank=self.fabric.global_rank
        world_size=self.fabric.world_size


        file_ext = ''.join(pathlib.Path(filename).suffixes)       
        if file_ext=='.json':
            with open(filename, "r") as f:  
                data = json.load(f)
        elif file_ext=='.json.gz':
            with gzip.open(filename, 'rt') as f:  
                data = json.load(f)  

        examples = []

        for example_idx in range(len(data)):

            if (example_idx % world_size)==global_rank:
                question = data[example_idx][self.question_column_1]
                question_2 = data[example_idx][self.question_column_2]

                if question_2!=None:
                    question = question + '\n' + question_2
                    
                curr_example = {
                    'question': question,
                    'answer': data[example_idx][self.answer_column]
                }
                examples.append(curr_example)

        print(len(data), global_rank, world_size, len(examples))


        return examples      


    def _get_content(self, question: str, index: int):
        return question
        

    def get_score(self, gt_answer, res_answer):
        return 0.0
        

