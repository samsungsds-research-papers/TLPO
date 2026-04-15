# -*- coding: utf-8 -*-

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
import pathlib
import gzip  


from abc import ABC, abstractmethod

from . import confusion_detector as cd



class BaseDataset(ABC):

    def __init__(self,
                 data_params,
                 model_type,
                 fabric,
                 tokenizer,
                 per_device_batch_size):

        
        # save arguments
        self.data_prams = data_params
        self.model_type = model_type
        self.fabric = fabric
        self.tokenizer = tokenizer
        self.per_device_batch_size = per_device_batch_size
        self.examples = None
        self.num_exemples = None
        self.batch_len = None
        self.logger = logging.getLogger()
        self.target_language = data_params['target_language']
        self.ignore_english = data_params['ignore_english']

        self.input_tok = None
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id

        self.tokenizer.padding_side = 'left'

        if self.model_type == 'qwen':
            self._make_chat = self._make_chat_qwen
        else:
            self._make_chat = self._make_chat_llama


    @abstractmethod                
    def _get_content(self, question: str, index: int):
        pass

        
    @abstractmethod                
    def get_score(self, gt_answer, res_answer):
        pass        
        

    def _read_lang_instruction_in_target(self, target_language):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'lang_instructions', f'{target_language}.json')

        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
                
        return data


    def get_consistency(self, res_answer):
        cp = self.get_confusion_point(res_answer)
        consistency = 1.0 if cp == -1 else 0.0
                
        return float(consistency)


    def get_confusion_point(self, res_answer):
        cp = cd.get_confusion_point(res_answer, self.target_language, self.ignore_english)

        return cp
        

    def check_response(self, res_answer):        
        resp_p, resp_f, line_p, line_f, word_p, word_f = cd.check_response(res_answer, self.target_language, self.ignore_english)        
                
        return resp_p, resp_f, line_p, line_f, word_p, word_f


    def _make_chat_llama(self, question: str, index: int) -> list:
        system_prompt = ""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": self._get_content(question, index),
            },
        ]
    
        return messages


    def _make_chat_qwen(self, question: str, index: int) -> list:
        messages = [
            {
                "role": "user",
                "content": self._get_content(question, index),
            },
        ]
    
        return messages        


    def get_batch_len(self):
        return self.batch_len

    def get_per_device_data_len(self):
        return self.batch_len * self.per_device_batch_size


    def tokenize_all(self):
        start_idx = 0
        end_idx = len(self.examples)
        
        question = [self._make_chat(self.examples[idx]['question'], idx) for idx in range(start_idx, end_idx)]
        answer = [self.examples[idx]['answer'] for idx in range(start_idx, end_idx)]        
        prompt = [self.tokenizer.apply_chat_template(
                       q,
                       tokenize=False,
                       add_generation_prompt=True,
                       enable_thinking=False,
                   ) for i, q in enumerate(question)]


        tokenizer_output = self.tokenizer(prompt, padding=True, return_tensors="pt")
        self.input_tok = {k: v.to(self.fabric.device) for k, v in tokenizer_output.items()}

        self.logger.info(f"len(input_ids): {self.input_tok['input_ids'].size(1)}")


    def shuffle(self):
        example_indices = np.random.permutation(len(self.examples))
        
        self.examples = [self.examples[idx] for idx in example_indices]

        self.input_tok['input_ids'] = self.input_tok['input_ids'][example_indices]
        self.input_tok['attention_mask'] = self.input_tok['attention_mask'][example_indices]
        

    def get_input_ids_len(self):
        return self.input_tok['input_ids'].size(1)

    def get_input_ids_dtype(self):
        return self.input_tok['input_ids'].dtype

    def get_input_atn_mask_dtype(self):
        return self.input_tok['attention_mask'].dtype
        
        
    def get_batch(self, batch_idx):        
        start_idx = batch_idx * self.per_device_batch_size
        end_idx = min(start_idx+self.per_device_batch_size, self.num_exemples)
        if start_idx >= end_idx:
            return None

        question = [self.examples[idx]['question'] for idx in range(start_idx, end_idx)]
        answer = [self.examples[idx]['answer'] for idx in range(start_idx, end_idx)]
        
        input_ids = self.input_tok['input_ids'][start_idx:end_idx]
        attention_mask = self.input_tok['attention_mask'][start_idx:end_idx]
        
        batch = {'input_ids': input_ids,
                 'attention_mask': attention_mask,
                 'question': question,
                 'answer': answer}

        return batch    


    def load_dataset(self, filepath, do_truncate=False, do_shuffle=True, debug_max_batch_len=None):
        self.logger.info(f"loading: {filepath}")

        ##################
        # Load
        self.examples = self._load_examples(filepath, do_truncate)
        
        self.num_exemples = len(self.examples)
        if do_truncate:            
            batch_len = self.num_exemples//self.per_device_batch_size
            self.batch_len = self.fabric.all_reduce(batch_len, reduce_op='min')
        else:
            batch_len = math.ceil(self.num_exemples/self.per_device_batch_size)
            gbatch_len = self.fabric.all_reduce(batch_len, reduce_op='sum')
            self.batch_len = math.ceil(gbatch_len/self.fabric.world_size)

        print(f'[rank {self.fabric.global_rank}] batch_len {batch_len} -> {self.batch_len}', )

            
        ##################
        # Tokenize
        self.tokenize_all()

        ##################
        # Shuffle        
        if do_shuffle:
            self.shuffle()

        ##################
        # Debug setting
        if debug_max_batch_len is not None:
            self.batch_len = min(self.batch_len, debug_max_batch_len)
            
        self.logger.info(f"data loaded, per_device_size={len(self.examples)}, batch_len={self.batch_len}, truncated={do_truncate}")


    def _load_examples(self,
                        filepath,
                        do_truncate):
                        
        file_ext = ''.join(pathlib.Path(filepath).suffixes)
       
        if file_ext=='.json':
            example_loader = self._load_json_examples        
        elif file_ext=='.json.gz':
            example_loader = self._load_json_gz_examples        
        elif file_ext=='.jsonl':
            example_loader = self._load_jsonl_examples
        elif file_ext=='.parquet':
            example_loader = self._load_parquet_examples
        else:
            raise ValueError("invalid file_ext: [{}]".format(file_ext))
            
        examples = example_loader(
            filename=filepath,
            do_truncate=do_truncate,
            per_device_batch_size=self.per_device_batch_size,
            global_rank=self.fabric.global_rank,
            world_size=self.fabric.world_size)            

        return examples                       
        
    
    def _load_json_examples(self,
                            filename,
                            do_truncate,
                            per_device_batch_size,
                            global_rank,
                            world_size):
        examples = []

        with open(filename, "r") as f:  
            data = json.load(f)

        for example_idx in range(len(data)):
            if (example_idx % world_size)==global_rank:
                curr_example = {
                    'question': data[example_idx][self.question_column],
                    'answer': data[example_idx][self.answer_column]
                }
                examples.append(curr_example)

        return examples           


    def _load_json_gz_examples(self,
                            filename,
                            do_truncate,
                            per_device_batch_size,
                            global_rank,
                            world_size):
        examples = []

        with gzip.open(filename, 'rt') as f:  
            data = json.load(f)  

        for example_idx in range(len(data)):
            if (example_idx % world_size)==global_rank:
                curr_example = {
                    'question': data[example_idx][self.question_column],
                    'answer': data[example_idx][self.answer_column]
                }
                examples.append(curr_example)

        return examples      


    def _load_jsonl_examples(self,
                            filename,
                            do_truncate,
                            per_device_batch_size,
                            global_rank,
                            world_size):
        examples = []

        for example_idx, line in enumerate(open(filename)):
            if (example_idx % world_size)==global_rank:
                curr_line = json.loads(line)
                curr_example = {
                    'question': curr_line[self.question_column],
                    'answer': curr_line[self.answer_column]
                }                
                examples.append(curr_example)

        self.num_exemples = len(examples)

        if do_truncate:
            batch_len = self.num_exemples//per_device_batch_size
            gbatch_len = self.fabric.all_reduce(batch_len, reduce_op='sum')
            self.batch_len = gbatch_len//world_size
        else:
            batch_len = math.ceil(self.num_exemples/per_device_batch_size)
            gbatch_len = self.fabric.all_reduce(batch_len, reduce_op='sum')
            self.batch_len = math.ceil(gbatch_len/world_size)
            
        return examples 


    def _load_parquet_examples(self,
                            filename,
                            do_truncate,
                            per_device_batch_size,
                            global_rank,
                            world_size):
        examples = []

        df = pd.read_parquet(filename)
        for example_idx in range(len(df)):
            if (example_idx % world_size)==global_rank:
                curr_example = {
                    'question': df.iloc[example_idx][self.question_column],
                    'answer': df.iloc[example_idx][self.answer_column]
                }
                examples.append(curr_example)

        return examples 


