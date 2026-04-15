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

import os
import torch
import numpy as np
import pandas as pd
import math
import logging
import shutil
import platform
import pdb
from abc import ABC, abstractmethod

from torch.optim import AdamW
import torch.nn.functional as F

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed import ReduceOp

from utils import get_result_folder, TimeEstimator

from dataset.dataset_creator import create_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Gemma3ForCausalLM

PLATFORM = platform.system()


class BaseTrainer(ABC):
    def __init__(self,
                 fabric,
                 train_data_params,
                 valid_data_params,
                 model_params,
                 optimizer_params,
                 train_params):

        # save arguments
        self.fabric = fabric
        self.train_data_params = train_data_params
        self.valid_data_params = valid_data_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.train_params = train_params

        # logger, dir
        self.logger = logging.getLogger()
        self.result_folder = get_result_folder()
        self.model_save_dir = os.path.join(self.result_folder, "checkpoint")    

        # model, optimizer
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.optimizer = None
        self.lr_schedule_fn = None
        self.accumulate_grad_batches = None
        self.per_device_update_batch_size = None

        self.start_epoch = 1
        self.step_idx = 0
        
        # data loader
        self.data_train = None
        self.data_valid = None
        self.max_length = None
        self.total_solutions = None
        
        # TimeEstimator
        self.time_estimator = TimeEstimator()

        # Healthy tag
        self._create_healthy_tag()
        

    @abstractmethod            
    def _get_per_device_gen_batch_size_of_train_data(self):
        pass


    @abstractmethod            
    def _train_epoch(self, epoch):
        pass


    def _create_healthy_tag(self):
        healthy_tag_path = os.path.join(self.result_folder, 'run.txt')
        with open(healthy_tag_path, 'w') as file:
            file.write(' ')
        

    
    def run(self):            
        self._log_info("-"*50)


        # init model and optimizer
        self._init_model_and_optimizer()

        
        # load dataset
        self.data_train = create_dataset(data_type=self.train_data_params['data_type'],
                                       data_params=self.train_data_params,
                                       model_type=self.model_params['model_type'],
                                       fabric=self.fabric, 
                                       tokenizer=self.tokenizer, 
                                       per_device_batch_size=self._get_per_device_gen_batch_size_of_train_data())
                                       
        self.data_valid = create_dataset(data_type=self.valid_data_params['data_type'],
                                       data_params=self.valid_data_params,
                                       model_type=self.model_params['model_type'],
                                       fabric=self.fabric, 
                                       tokenizer=self.tokenizer, 
                                       per_device_batch_size=self.train_params['per_device_gen_batch_size'])
        
        self.data_train.load_dataset(self.train_data_params['data_file'], 
                                     do_truncate=True, 
                                     do_shuffle=True, 
                                     debug_max_batch_len=self.train_data_params['debug_batch_len'])
                                     
        self.data_valid.load_dataset(self.valid_data_params['data_file'], 
                                     do_truncate=False, 
                                     do_shuffle=False,
                                     debug_max_batch_len=self.valid_data_params['debug_batch_len'])

        self.total_solutions = self.data_train.get_per_device_data_len()*self.fabric.world_size*self.train_params['num_concurrent_solutions']
        self.fabric.barrier()

         
        # train
        self._train()
        self._log_info("----- Complete -----")


    def _init_model_and_optimizer(self):

        ####################
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_params['model_path'], padding_side='left')


        CausalLM = Gemma3ForCausalLM if 'gemma' in self.model_params['model_type'] else AutoModelForCausalLM

        ####################
        # load model       
        self.model = CausalLM.from_pretrained(self.model_params['model_path'])
        self._log_info("="*50)
        self._log_info(f"hf checkpoint loaded from {self.model_params['model_path']}")
        self._log_info("="*50)

        model_size = sum(t.numel() for t in self.model.parameters())    
        self._log_info("*"*50)
        self._log_info(f"model size: {model_size/1000**3:.3f}B parameters")         
        self._log_info("*"*50)


        self.ref_model = CausalLM.from_pretrained(self.model_params['model_path'])

        ####################
        # initialize optimizer
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.optimizer_params['lr'],
                                weight_decay=self.optimizer_params['weight_decay'],
                                betas=(self.optimizer_params['beta1'], self.optimizer_params['beta2']),
                                foreach=False)

        ####################
        # fabric setup
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.ref_model = self.fabric.setup(self.ref_model)
        self.model.mark_forward_method("generate")
        
        ####################
        # epoch, step_idx
        self.start_epoch = self.train_params['start_epoch']
        self.step_idx = self.train_params['step_idx']

        self._log_info(f"step_idx: {self.step_idx}")
        self._log_info(f"start epoch: {self.start_epoch }")
        self._log_info(f"num_epoch: {self.train_params['num_epoch']}")
        
        torch.cuda.empty_cache()
        


    def _init_lr_scheduler(self, epoch):
        ####################
        # lr scheduler
        total_batch_len = self.data_train.get_batch_len()
        lr_decay_per_epoch = self.optimizer_params['lr_decay_per_epoch']
        lr = self.optimizer_params['lr'] * (lr_decay_per_epoch**(epoch-1))
        end_lr = self.optimizer_params['end_lr'] * (lr_decay_per_epoch**(epoch-1))

        self._log_info(f"lr: {lr}, end_lr: {end_lr}, total_steps for lr scheduler: {total_batch_len}")
        
        self.lr_schedule_fn = self._get_cosine_lr_decay_fn(
            total_steps=total_batch_len,
            warmup_steps=int(total_batch_len*self.optimizer_params['warmup_rate']),
            learning_rate=lr,
            end_learning_rate=end_lr)
        
        

    def _get_cosine_lr_decay_fn(self,
                               total_steps,
                               warmup_steps,
                               learning_rate,
                               end_learning_rate):
        def cosine_with_warmup_lr(step):
            if step < warmup_steps:
                return max(1e-8, learning_rate * step / warmup_steps)
            elif step > total_steps:
                return end_learning_rate
    
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return end_learning_rate + coeff * (learning_rate - end_learning_rate)
    
        return cosine_with_warmup_lr
        

    def _train(self):
        ####################
        # accumulate_grad_batches variable
        self.per_device_update_batch_size = int(self.train_params['update_batch_size'] // self.fabric.world_size)
        self.accumulate_grad_batches = int(self.per_device_update_batch_size // self.train_params['per_device_forward_batch_size'])
        assert self.accumulate_grad_batches > 0
        
        assert(self.accumulate_grad_batches*self.train_params['per_device_forward_batch_size']*self.fabric.world_size\
                    ==self.train_params['update_batch_size'])
        self._log_info(f"accumulate_grad_batches: {self.accumulate_grad_batches}")
        
                
        ####################
        # variables for time estimator
        self.global_start_batch_idx = 0
        self.global_end_batch_idx = self.data_train.get_batch_len() * self.train_params['num_epoch']

        self._log_info(f"global_start_batch_idx: {self.global_start_batch_idx}, global_end_batch_idx: {self.global_end_batch_idx}")


        ####################
        # save before train
        if self.train_params['save_test']:
            self._save_model(-1)   # save test

        ####################
        # eval before train
        if self.train_params['eval_before_train']:
            self._evaluate_accuracy(0)


        ####################
        # init replay buffer
        local_max_length = self.data_train.get_input_ids_len() + self.train_data_params['max_gen_tokens']
        self.max_length = self.fabric.all_reduce(local_max_length, reduce_op=ReduceOp.MAX)
        

        ####################
        # reset time estimator    
        self.time_estimator.reset()


        for epoch in range(self.start_epoch, self.train_params['num_epoch']+1):
            self._log_info("")
            self._log_info("="*100)
            self._log_info(f"train epoch {epoch}, step_idx {self.step_idx}")

            # init lr_scheduler, lr_scheduler should be initialized after loading dataset.
            self._init_lr_scheduler(epoch)
            self.fabric.barrier()

            # update reference mode
            self._log_info(f"update reference model")
            self._copy_model(self.ref_model, self.model)

            # dataset            
            self.data_train.shuffle()
            
            # train epoch
            self.fabric.barrier()
            self._train_epoch(epoch)
        

            # evaluation
            self._evaluate_accuracy(self.step_idx)
            self.time_estimator.print_est_time(self.global_end_batch_idx-self.global_start_batch_idx, self.global_end_batch_idx-self.global_start_batch_idx)

            # save model
            if epoch % self.train_params['save_interval'] == 0 or epoch==self.train_params['num_epoch']:
                self._save_model(epoch)



    def _get_sel_log_probs(self, logits, ids_shift):
        # logits.shape: (batch, sequence-1, vocab)
        # ids_shift.shape: (batch, sequence-1)

        log_probs = F.log_softmax(logits, dim=2)
        # shape: (batch, sequence-1, vocab)
        
        sel_log_probs = torch.gather(log_probs, 2, ids_shift[:, :, None]).squeeze(2)
        # shape: (batch, sequence-1)

        return sel_log_probs

                     

    def _allgather_and_shard(self, tensor):
        device = tensor.device
        rank = self.fabric.global_rank
        world_size = self.fabric.world_size

        # get bach_size of each tensor
        local_n = torch.tensor([tensor.size(0)], device=device)
        gathered_ns = self.fabric.all_gather(local_n)                # shape: (world_size, 1)
        ns = gathered_ns.flatten()                              # shape: (world_size,)

        # get max batch_size and pad
        max_n = int(ns.max().item())
        if tensor.size(0) < max_n:
            pad_n = max_n - tensor.size(0)
            pad_size = pad_n, *tensor.shape[1:]
            pad_tensor = torch.zeros(pad_size, device=device, dtype=tensor.dtype)
            padded = torch.cat([tensor, pad_tensor], dim=0)
        else:
            padded = tensor


        # all_gather
        gathered = self.fabric.all_gather(padded)                    # shape: (world_size, max_n, k)
        
        # remove pad
        chunks = []
        for i, ni in enumerate(ns.tolist()):
            chunks.append(gathered[i, :ni])

        global_tensor = torch.cat(chunks, dim=0)

        # shard
        total_len = global_tensor.size(0)
        per_rank = total_len // world_size
        start = rank * per_rank
        end = start + per_rank if rank < world_size - 1 else total_len
        local_shard = global_tensor[start:end]
        
        return local_shard
        

    @torch.no_grad()  
    def _evaluate_accuracy(self, step):
        self.model.eval()

        eval_batch_len = self.data_valid.get_batch_len()
        log_interval = 1
        next_log_step = log_interval

        score_list = []
        rp_list = []
        rf_list = []
        lp_list = []
        lf_list = []
        wp_list = []
        wf_list = []

        log_question_list = []
        log_answer_list = []
        log_response_list = []

        for batch_idx in range(eval_batch_len):
            batch = self.data_valid.get_batch(batch_idx)      
            if batch==None:
                break

            questions = batch["question"]
            answers = batch["answer"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            gen_max_length = input_ids.shape[1]+self.valid_data_params['max_gen_tokens']

            generated_ids = self.model.generate(inputs=input_ids, 
                                            attention_mask=attention_mask,
                                            do_sample=False, 
                                            temperature=None,
                                            top_p=None,
                                            top_k=None,
                                            max_length=gen_max_length, 
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            use_cache=True)
            response = self.tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
            # shape: (batch, response)

            del generated_ids
            torch.cuda.empty_cache()

            #print(response[0])

            for i, res in enumerate(response):
                ans = answers[i]
                que = questions[i]
                score = self.data_valid.get_score(ans, res)
                rp, rf, lp, lf, wp, wf = self.data_valid.check_response(res)
                score_list.append(score)
                rp_list.append(rp)
                rf_list.append(rf)
                lp_list.append(lp)
                lf_list.append(lf)
                wp_list.append(wp)
                wf_list.append(wf)

                log_question_list.append(que)
                log_answer_list.append(ans)
                log_response_list.append(res)
                
                            
            if batch_idx >= next_log_step:
                score_mean = np.array(score_list).mean()       # per rank

                rp_sum = np.array(rp_list).sum()
                rf_sum = np.array(rf_list).sum()
                lp_sum = np.array(lp_list).sum()
                lf_sum = np.array(lf_list).sum()
                wp_sum = np.array(wp_list).sum()
                wf_sum = np.array(wf_list).sum()
                                
                rfr = 100.0*rf_sum/(rp_sum+rf_sum) if (rp_sum+rf_sum)>0 else 0.0
                lfr = 100.0*lf_sum/(lp_sum+lf_sum) if (lp_sum+lf_sum)>0 else 0.0
                wfr = 100.0*wf_sum/(wp_sum+wf_sum) if (wp_sum+wf_sum)>0 else 0.0

                self._log_info(f"\t[rank={self.fabric.global_rank}] evaluate ({batch_idx}/{eval_batch_len}), "\
                                 f"score_mean: {score_mean:.2f}, rfr: {rfr:.4f}, lfr: {lfr:.4f}, wfr: {wfr:.4f}")
                next_log_step += log_interval

        if self.train_params['save_response_csv']:
            log_csv_dir = os.path.join(self.result_folder, 'response_csv', f"step_{step}")

            if self.fabric.global_rank == 0:
                os.makedirs(log_csv_dir, exist_ok = True)

            self.fabric.barrier()                
            
            log_cvs_name = os.path.join(log_csv_dir, f"rank{self.fabric.global_rank}.csv")
            log_to_save_df = pd.DataFrame({
                "question": log_question_list,
                "response": log_response_list,
                "answer": log_answer_list,
                "score": score_list
            })
            log_to_save_df.to_csv(log_cvs_name, index=False, encoding='utf-8', mode='w')
                
        
        score_sum = torch.tensor(score_list).sum().to(self.fabric.device)
        rp_sum = torch.tensor(rp_list).sum().to(self.fabric.device)
        rf_sum = torch.tensor(rf_list).sum().to(self.fabric.device)
        lp_sum = torch.tensor(lp_list).sum().to(self.fabric.device)
        lf_sum = torch.tensor(lf_list).sum().to(self.fabric.device)
        wp_sum = torch.tensor(wp_list).sum().to(self.fabric.device)
        wf_sum = torch.tensor(wf_list).sum().to(self.fabric.device)
        score_cnt = torch.tensor(len(score_list)).to(self.fabric.device)
        
        
        gscore_sum = self.fabric.all_reduce(score_sum, reduce_op='sum').cpu().item()
        grp_sum = self.fabric.all_reduce(rp_sum, reduce_op='sum').cpu().item()
        grf_sum = self.fabric.all_reduce(rf_sum, reduce_op='sum').cpu().item()
        glp_sum = self.fabric.all_reduce(lp_sum, reduce_op='sum').cpu().item()
        glf_sum = self.fabric.all_reduce(lf_sum, reduce_op='sum').cpu().item()
        gwp_sum = self.fabric.all_reduce(wp_sum, reduce_op='sum').cpu().item()
        gwf_sum = self.fabric.all_reduce(wf_sum, reduce_op='sum').cpu().item()
        gscore_cnt = self.fabric.all_reduce(score_cnt, reduce_op='sum').cpu().item()

        gscore_mean = gscore_sum/gscore_cnt
        grfr = 100.0*grf_sum/(grp_sum+grf_sum) if (grp_sum+grf_sum)>0 else 0.0
        glfr = 100.0*glf_sum/(glp_sum+glf_sum) if (glp_sum+glf_sum)>0 else 0.0
        gwfr = 100.0*gwf_sum/(gwp_sum+gwf_sum) if (gwp_sum+gwf_sum)>0 else 0.0

        self._log_info(f"eval [step: {step}, accuracy: {100.0*gscore_mean:.4f}%]")
        self._log_info(f"eval [step: {step}, rfr: {grfr:.4f}%, lfr: {glfr:.4f}%, wfr: {gwfr:.4f}%]")      
                    
        return gscore_mean


    def _get_grad_norm(self, model):
        square_sum = 0.
        for param in model.parameters():
            if param.grad is not None:
                square_sum += param.grad.detach().data.norm(2).item() ** 2
        return square_sum ** 0.5


    @torch.no_grad()  
    def _copy_model(self, model_dst, model_src):
        for src_param, dst_param in zip(model_src.parameters(), model_dst.parameters()):
            dst_param.copy_(src_param)
            

    def _save_model(self, epoch):
        pretrained_save_dir = f"{self.model_save_dir}_{epoch}"
        self.fabric.barrier()
                
        if not isinstance(self.fabric.strategy, FSDPStrategy):
            self._log_info(f"Error: _save_model() for stratege {self.fabric.strategy} is not supported.")
            self._log_info("Error: saving model failed")
            return

        ###############################
        # save hf model
        save_policy = FullStateDictConfig(
            offload_to_cpu=(self.fabric.world_size > 1), rank0_only=True)

        with FSDP.state_dict_type(
                self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=save_policy):
            state_dict = self.model._forward_module.state_dict()

        if self.fabric.global_rank == 0:
            os.makedirs(pretrained_save_dir, exist_ok=True)
            self.tokenizer.save_pretrained(pretrained_save_dir)
            self.model.module.save_pretrained(
                pretrained_save_dir, state_dict=state_dict, safe_serialization=False)

        self.logger.info(f"{pretrained_save_dir} saved")


        ###############################
        # remove -1 model
        if epoch==-1 and self.fabric.global_rank == 0:
            shutil.rmtree(pretrained_save_dir)            
            self.logger.info(f"{pretrained_save_dir} removed")

        ###############################
        # logging 
        if epoch>=0:
            self.logger.info(f"next_epoch: {epoch+1}, next_step_idx: {self.step_idx}")

        self.fabric.barrier()
            

    def _log_info(self, msg):
        if self.fabric.global_rank==0:
           self.logger.info(msg)
        

