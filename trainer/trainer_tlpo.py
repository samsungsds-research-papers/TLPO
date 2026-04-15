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
import numpy as np
import platform
import pdb

import torch.nn.functional as F

from trainer.trainer_base import BaseTrainer

PLATFORM = platform.system()


class Trainer_TLPO(BaseTrainer):
    def __init__(self,
                 fabric,
                 train_data_params,
                 valid_data_params,
                 model_params,
                 optimizer_params,
                 train_params):

        super().__init__(fabric,
                 train_data_params,
                 valid_data_params,
                 model_params,
                 optimizer_params,
                 train_params)


    
    
    def _get_per_device_gen_batch_size_of_train_data(self):
        assert self.train_params['per_device_gen_batch_size'] >= self.train_params['num_concurrent_solutions']
        return self.train_params['per_device_gen_batch_size'] // self.train_params['num_concurrent_solutions']
           

    def _train_epoch(self, epoch):
    
        self.model.train()
        self.ref_model.eval()
    
        batch_len = self.data_train.get_batch_len()     # number of batch data        
    
        eval_interval_batch = int(max(1, batch_len / self.train_params['eval_num_in_epoch']))
        self._log_info(f"eval_interval_batch={eval_interval_batch}")
        
        # logging
        train_score_list = []      
        
        # memory
        mem1, mem2 = self._create_memory_buffers()
        
        
        for batch_idx in range(batch_len):
            batch = self.data_train.get_batch(batch_idx)                
    
            train_score_list_batch = self._generate_train_samples(batch, mem1, mem2, epoch, batch_idx, batch_len)
            train_score_list.extend(train_score_list_batch)  
    
    
            ##############################################################################
            # update model
            #
            self._update_model(mem2, epoch, batch_idx, batch_len)
    
    
            ##############################################################################
            # Logging           
            # train score
            if len(train_score_list)>0:
                self._log_info(f"{{epoch: {epoch}/{self.train_params['num_epoch']}, steps: {self.step_idx}, "\
                                 f"batchs: {batch_idx}/{batch_len}, "\
                                 f"score: {np.array(train_score_list).mean():.4f}}}")
                train_score_list = []                                     
    
    
            ##############################################################################
            # evaluation
            #
            eiv = eval_interval_batch
            if batch_idx < (batch_len // 10):
                eiv = max(1, eiv // 4)
            elif batch_idx < (batch_len // 5):
                eiv = max(1, eiv // 2)
                
            if batch_idx>0 and (batch_idx % eiv)==0:
                # evaluation
                self._evaluate_accuracy(self.step_idx)
            
                # time estimation
                batch_idx_total = (epoch-self.start_epoch)*batch_len + batch_idx                
                self.time_estimator.print_est_time(batch_idx_total-self.global_start_batch_idx, self.global_end_batch_idx-self.global_start_batch_idx)
    
                    
    def _create_memory_buffers(self):
        ids_dtype = self.data_train.get_input_ids_dtype()
        atn_dtype = self.data_train.get_input_atn_mask_dtype()        
        
        mem1_sequence_ids = torch.zeros((0, self.max_length), dtype=ids_dtype, device=self.fabric.device)
        mem1_seq_attention_mask = torch.zeros((0, self.max_length), dtype=atn_dtype, device=self.fabric.device)
        mem1_update_pos = torch.zeros((0, ), dtype=ids_dtype, device=self.fabric.device)
        mem1_cp_token_idx = torch.zeros((0, ), dtype=torch.int64, device=self.fabric.device)
        
        mem2_sequence_ids = torch.zeros((0, self.max_length), dtype=ids_dtype, device=self.fabric.device)
        mem2_seq_attention_mask = torch.zeros((0, self.max_length), dtype=atn_dtype, device=self.fabric.device)
        mem2_update_pos = torch.zeros((0, ), dtype=ids_dtype, device=self.fabric.device)
        mem2_advantage = torch.zeros((0, ), dtype=torch.float, device=self.fabric.device)
    
        mem1 = {
            'sequence_ids': mem1_sequence_ids,
            'seq_attention_mask': mem1_seq_attention_mask,
            'update_pos': mem1_update_pos,
            'cp_token_idx': mem1_cp_token_idx,
        }
    
        mem2 = {
            'sequence_ids': mem2_sequence_ids,
            'seq_attention_mask': mem2_seq_attention_mask,
            'update_pos': mem2_update_pos,
            'advantage': mem2_advantage,
        }
    
        return mem1, mem2
    
        
    def _generate_train_samples(self, batch, mem1, mem2, epoch, batch_idx, batch_len):
        num_token_extension = self.train_params['num_token_extension']
    
        train_score_list_batch = []
        
        ##############################
        # generate experiences
        with torch.no_grad():
            sequence_ids, seq_attention_mask, update_pos, consistency, cp_token_idx = self._generate_solutions(batch)
            # sequence_ids.shape: (per_device_gen_batch_size, sequence)
            # seq_attention_mask.shape: (per_device_gen_batch_size, sequence)
            # update_pos.shape: (per_device_gen_batch_size, )
            # consistency.shape: (per_device_gen_batch_size, )
            # cp_token_idx.shape: (per_device_gen_batch_size, ) 
            # cp_token_idx : token index of the first confusion point
        
        # train score log
        consistency_all=consistency.mean()
        gconsistency_all = self.fabric.all_reduce(consistency_all, reduce_op='mean').cpu().item()
        train_score_list_batch.append(gconsistency_all)
        
        
        valid = (consistency==0)
        
        # store data to replay buffer
        if valid.sum() > 0:
            mem1['sequence_ids'] = torch.cat((mem1['sequence_ids'], sequence_ids[valid, :]), dim=0)
            mem1['seq_attention_mask'] = torch.cat((mem1['seq_attention_mask'], seq_attention_mask[valid, :]), dim=0)
            mem1['update_pos'] = torch.cat((mem1['update_pos'], update_pos[valid]), dim=0)
            mem1['cp_token_idx'] = torch.cat((mem1['cp_token_idx'], cp_token_idx[valid]), dim=0)
        
        
        
        ##############################################################################
        # generate alternative tokens for unintended language output
        #
        # check number of replay data
        num_replay = mem1['sequence_ids'].size(0)
        local_n = torch.tensor([num_replay], device=self.fabric.device)
        gathered_ns = self.fabric.all_gather(local_n)                # shape: (world_size, 1)
        ns = gathered_ns.flatten()                              # shape: (world_size,)
        sum_num_exp = ns.sum().item()
        
        self.logger.info(f"epoch: {epoch}/{self.train_params['num_epoch']}, steps: {self.step_idx}, "\
                         f"batchs: {batch_idx}/{batch_len}, gconsistency_all: {gconsistency_all}, global_1st_replay_buffer_size: {sum_num_exp}")
        
        mem1_size = sum_num_exp//self.fabric.world_size
        
        # gather and shard 1st replay data                
        if mem1_size > 0:
            mem1['sequence_ids'] = self._allgather_and_shard(mem1['sequence_ids'])                
            mem1['seq_attention_mask'] = self._allgather_and_shard(mem1['seq_attention_mask'])
            mem1['update_pos'] = self._allgather_and_shard(mem1['update_pos'])
            mem1['cp_token_idx'] = self._allgather_and_shard(mem1['cp_token_idx'])
        
        
        for bidx in range(0, mem1_size):
            with torch.no_grad():
                sub_sequence_ids, sub_score, sub_probs = self._generate_multiple_next_tokens(mem1['sequence_ids'][bidx].unsqueeze(0), 
                                                                                    mem1['seq_attention_mask'][bidx].unsqueeze(0),
                                                                                    mem1['cp_token_idx'][bidx], 
                                                                                    num_token_extension,
                                                                                    4)
        
            # sub_sequence_ids.shape: (num_concurrent_solutions, sequence)
            # sub_score.shape: (num_concurrent_solutions, )
            # sub_probs.shape: (num_concurrent_solutions, )
        
            advantage = self._get_advantage(sub_sequence_ids, sub_score, sub_probs)
            # advantage.shape: (num_concurrent_solutions, )
        
            # store data to replay buffer
            if torch.any(sub_probs != sub_probs[0]):
                sub_attention_mask = mem1['seq_attention_mask'][bidx][None, :].expand(num_token_extension, -1)
                sub_update_pos = mem1['update_pos'][bidx][None].expand(num_token_extension)
                                        
                mem2['sequence_ids'] = torch.cat((mem2['sequence_ids'], sub_sequence_ids), dim=0)
                mem2['seq_attention_mask'] = torch.cat((mem2['seq_attention_mask'], sub_attention_mask), dim=0)
                mem2['update_pos'] = torch.cat((mem2['update_pos'], sub_update_pos), dim=0)
                mem2['advantage'] = torch.cat((mem2['advantage'], advantage), dim=0)
        
        
        ##############################
        # remove used memory
        if mem1_size > 0:
            mem1['sequence_ids'] = mem1['sequence_ids'][mem1_size:]
            mem1['seq_attention_mask'] = mem1['seq_attention_mask'][mem1_size:]
            mem1['update_pos'] = mem1['update_pos'][mem1_size:]
            mem1['cp_token_idx'] = mem1['cp_token_idx'][mem1_size:]
            torch.cuda.empty_cache()
    
        return train_score_list_batch
    
        
    def _update_model(self, mem2, epoch, batch_idx, batch_len):
        ppo_reuse = self.optimizer_params['ppo_reuse']        
        update_batch_size = self.train_params['update_batch_size']
        per_device_forward_batch_size = self.train_params['per_device_forward_batch_size']
    
        while True:
            ##############################
            # check number of replay data
            num_replay = mem2['sequence_ids'].size(0)
            local_n = torch.tensor([num_replay], device=self.fabric.device)
            gathered_ns = self.fabric.all_gather(local_n)                # shape: (world_size, 1)
            ns = gathered_ns.flatten()                              # shape: (world_size,)
            min_num_exp = ns.min().item()
            sum_num_exp = min_num_exp * self.fabric.world_size
        
            self.logger.info(f"epoch: {epoch}/{self.train_params['num_epoch']}, steps: {self.step_idx}, "\
                             f"batchs: {batch_idx}/{batch_len}, global_2nd_replay_buffer_size: {sum_num_exp} ({ns.tolist()})")
        
            if sum_num_exp < update_batch_size:
                break
                                                   
        
            for reuse_cnt in range(ppo_reuse):
                ##############################
                # forward/backward
                for acc_idx in range(self.accumulate_grad_batches):
                    # is accumulating                
                    is_accumulating = False if acc_idx==(self.accumulate_grad_batches-1) else True                
        
                    # forward/backward
                    s_idx = acc_idx * per_device_forward_batch_size
                    e_idx = s_idx + per_device_forward_batch_size
        
                    with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                        with torch.no_grad():
                            ref_outputs = self.ref_model(input_ids=mem2['sequence_ids'][s_idx:e_idx], 
                                                 labels=None,
                                                 attention_mask=mem2['seq_attention_mask'][s_idx: e_idx],
                                                 use_cache=False,
                                                 ignore_index=self.tokenizer.pad_token_id)
        
                            ref_sel_log_probs = self._get_sel_log_probs(logits=ref_outputs.logits, 
                                                                        seq_ids=mem2['sequence_ids'][s_idx:e_idx],
                                                                        update_pos=mem2['update_pos'][s_idx:e_idx])
        
                            del ref_outputs
                            torch.cuda.empty_cache()
        
                            
                        # forward, backward
                        old_sel_log_probs = None
        
                        for reuse_cnt in range(ppo_reuse):
                            outputs = self.model(input_ids=mem2['sequence_ids'][s_idx:e_idx], 
                                                 labels=None,
                                                 attention_mask=mem2['seq_attention_mask'][s_idx: e_idx],
                                                 use_cache=False,
                                                 ignore_index=self.tokenizer.pad_token_id)
        
                            loss, oslp = self._loss_function(logits=outputs.logits,
                                                  ref_sel_log_probs=ref_sel_log_probs,
                                                  old_sel_log_probs=old_sel_log_probs,
                                                  seq_ids=mem2['sequence_ids'][s_idx:e_idx],
                                                  update_pos=mem2['update_pos'][s_idx:e_idx],
                                                  advantage=mem2['advantage'][s_idx:e_idx])                                         
        
                            if old_sel_log_probs == None:
                                old_sel_log_probs = oslp.clone()
                        
                            self.fabric.backward(loss / self.accumulate_grad_batches)
        
                        del outputs
                        torch.cuda.empty_cache()
        
              
                ##############################
                # update
                
                # lr
                lr = self.lr_schedule_fn(batch_idx)                    
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                
                # update parameters
                grad_norm = self._get_grad_norm(model=self.model)
                self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.optimizer_params['grad_norm_clip'])
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # increase step count
                self.step_idx += 1
        
        
            ##############################
            # remove used memory
            mem2['sequence_ids'] = mem2['sequence_ids'][self.per_device_update_batch_size:]
            mem2['seq_attention_mask'] = mem2['seq_attention_mask'][self.per_device_update_batch_size:]
            mem2['update_pos'] = mem2['update_pos'][self.per_device_update_batch_size:]
            mem2['advantage'] = mem2['advantage'][self.per_device_update_batch_size:]
            torch.cuda.empty_cache()
    
    
    
    @torch.no_grad()  
    def _generate_solutions(self, batch):
    
        input_ids = batch['input_ids']                
        attention_mask = batch['attention_mask']
        # shape: (per_device_gen_batch_size//num_concurrent_solutions, prompt+padding)
        
        
        temperature = self.train_params['temperature']
        max_gen_tokens = self.train_data_params['max_gen_tokens']

        num_concurrent_solutions = self.train_params['num_concurrent_solutions']
        concurrent_batch_size = input_ids.size(0) * num_concurrent_solutions

        # expand input
        input_ids_ext = input_ids[:, None, :].expand(-1, num_concurrent_solutions, -1)\
                                             .reshape(concurrent_batch_size, -1)
        attention_mask_ext = attention_mask[:, None, :].expand(-1, num_concurrent_solutions, -1)\
                                                       .reshape(concurrent_batch_size, -1)
        # shape: (per_device_gen_batch_size, prompt+padding)
        

        # generate
        gen_max_length = input_ids_ext.shape[1]+max_gen_tokens            
        
        gen_outputs = self.model.generate(inputs=input_ids_ext, 
                                        attention_mask=attention_mask_ext,
                                        do_sample=True,
                                        temperature=temperature,
                                        max_length=gen_max_length, 
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        use_cache=True,
                                        return_dict_in_generate=True)



        # decode and re-encode to align the confusion position with token index
        response = self.tokenizer.batch_decode(gen_outputs.sequences, skip_special_tokens=True)    
        tokenizer_output = self.tokenizer(response, padding=True, return_tensors="pt")
        sequence_ids = tokenizer_output['input_ids'].to(self.fabric.device)
        attention_mask = tokenizer_output['attention_mask'].to(self.fabric.device)

        response_gen_only = self.tokenizer.batch_decode(gen_outputs.sequences[:, input_ids_ext.shape[1]:], skip_special_tokens=True)

        # delete memory for model.generate()
        del gen_outputs
        torch.cuda.empty_cache()
        

        # pad sequence ids
        if sequence_ids.size(1) < self.max_length:
            batch_size = sequence_ids.size(0)
            pad_len = self.max_length - sequence_ids.size(1)

            pad_ids = torch.full((batch_size, pad_len), 
                                self.tokenizer.pad_token_id, 
                                dtype=sequence_ids.dtype, 
                                layout=sequence_ids.layout, 
                                device=sequence_ids.device)

            sequence_ids = torch.cat((sequence_ids, pad_ids), dim=1)
            

        # init update mask
        update_pos = torch.full_like(sequence_ids[:, 0], -1)

        # get scores and update mask
        consistency_score_list = []
        cp_token_idx = []

        for idx, (res_a, res_g) in enumerate(zip(response, response_gen_only)):
            res_start_pos = res_a.rfind(res_g)
        
            cp_gen_only = self.data_train.get_confusion_point(res_a[res_start_pos:])
            consistency = 1.0 if cp_gen_only == -1 else 0.0
            consistency_score_list.append(consistency)

            if consistency == 1.0:
                cp_token_idx.append(-1)
            else:
                cp = cp_gen_only + res_start_pos
                cp_ti = tokenizer_output.char_to_token(idx, cp)

                self.logger.info(f"response : \n{res_a[cp-16: cp+16]}")
                self.logger.info(f"\t[{res_a[cp: cp+16]}]")
                
                update_pos[idx] = cp_ti-1
                cp_token_idx.append(cp_ti)
                                        
        consistency_score_pt = torch.tensor(consistency_score_list, device=self.fabric.device)
        cp_token_idx = torch.tensor(cp_token_idx, device=self.fabric.device)
        
        # make sequence attention mask
        sequence_attention_mask = torch.ones_like(sequence_ids)
        sequence_attention_mask[:, :attention_mask.shape[1]] = attention_mask
        sequence_attention_mask[sequence_ids == self.tokenizer.pad_token_id] = 0
        

        return sequence_ids, sequence_attention_mask, update_pos, consistency_score_pt, cp_token_idx

    
    
    @torch.no_grad()  
    def _generate_multiple_next_tokens(self, in_sequence_ids, in_attention_mask, cp_ti, num_token_extension, gen_len):
        # sequence_ids.shape: (1, sequence)
        # cp_ti: scalar, token index of the first confusion point
        # num_concurrent_solutions: scalar, number of multiple solutions to generate
        # gen_len: scalar, generation length
    
    
        temperature = self.train_params['temperature']
    
        input_ids = in_sequence_ids[:, :cp_ti]     
        input_attention_mask = in_attention_mask[:, :cp_ti]
        # shape: (1, cp_ti)
    
        # generate 1 token
        gen_outputs_1st = self.model.generate(inputs=input_ids, 
                                        attention_mask=input_attention_mask,
                                        do_sample=True,
                                        temperature=temperature,
                                        max_length=cp_ti+1, 
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        use_cache=True,
                                        return_dict_in_generate=True,
                                        output_logits=True)
    
        last_logits = gen_outputs_1st.logits[0].clone()
        del gen_outputs_1st
    
    
        # get next tokens
        last_probs = F.softmax(last_logits, dim=-1)
        top_values, top_indices = torch.topk(last_probs, num_token_extension, dim=1)
    
        top_indices = top_indices.squeeze(0).unsqueeze(1)
        top_values = top_values.squeeze(0)
    
        ones = torch.ones(input_attention_mask.size(0), 1, device=input_attention_mask.device, dtype=input_attention_mask.dtype)
    
        
        input_ids_ext = torch.concat([input_ids.expand(num_token_extension, -1), top_indices], dim=1)
        # shape: (num_concurrent_solutions, cp_ti+1)
    
        input_attention_mask_ext = torch.concat([input_attention_mask, ones], dim=1).expand(num_token_extension, -1)
        # shape: (num_concurrent_solutions, cp_ti+1)
    
    
        # generate
        gen_max_length = cp_ti+gen_len            
        
        gen_outputs = self.model.generate(inputs=input_ids_ext, 
                                        attention_mask=input_attention_mask_ext,
                                        do_sample=False,
                                        temperature=None,
                                        top_p=None,
                                        top_k=None,                                        
                                        max_length=gen_max_length, 
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        use_cache=True,
                                        return_dict_in_generate=True)
    
    
        # pad sequence ids
        if gen_outputs.sequences.size(1) < self.max_length:
            batch_size = gen_outputs.sequences.size(0)
            pad_len = self.max_length - gen_outputs.sequences.size(1)
    
            pad_ids = torch.full((batch_size, pad_len), 
                                self.tokenizer.pad_token_id, 
                                dtype=gen_outputs.sequences.dtype, 
                                layout=gen_outputs.sequences.layout, 
                                device=gen_outputs.sequences.device)
    
            sequence_ids = torch.cat((gen_outputs.sequences, pad_ids), dim=1)
        else:
            sequence_ids = gen_outputs.sequences.clone()
    
            
        # delete memory for model.generate()
        del gen_outputs
        torch.cuda.empty_cache()
    
        
        # get response
        response = self.tokenizer.batch_decode(sequence_ids[:, cp_ti:], skip_special_tokens=True)    
    
        
        consistency_score_list = []
        for idx, res_a in enumerate(response):
            consistency = self.data_train.get_consistency(res_a)
            score = 1.0 if consistency == 1.0 else -1.0
            consistency_score_list.append(score)
    
        consistency_score_pt = torch.tensor(consistency_score_list, device=self.fabric.device)
    
    
        return sequence_ids, consistency_score_pt, top_values
        
    
    def _get_sel_log_probs(self, logits, seq_ids, update_pos):
        # logits.shape: (batch, sequence, vocab)
        # ids_shift.shape: (batch, sequence)
    
        vocab_size = logits.size(2)
        logits_valid = torch.gather(logits, 1, update_pos[:, None, None].expand(-1, -1, vocab_size)).squeeze(1)
        # shape: (batch, vocab)
        
        seq_ids_shift = torch.gather(seq_ids[:, 1:], 1, update_pos[:, None]).squeeze(1)
        # shape: (batch, )
    
        log_probs = F.log_softmax(logits_valid, dim=1)
        # shape: (batch, vocab)
    
        sel_log_probs = torch.gather(log_probs, 1, seq_ids_shift[:, None]).squeeze(1)
        # shape: (batch, )
    
        return sel_log_probs


    def _get_advantage(self, sub_sequence_ids, sub_score, sub_probs):
        # sub_sequence_ids.shape: (num_concurrent_solutions, sequence)
        # sub_score.shape: (num_concurrent_solutions, )
        # sub_probs.shape: (num_concurrent_solutions, )
        assert sub_score.size(0) > 1
        
        sub_probs = sub_probs.detach()

        mu = (sub_probs * sub_score).sum(dim=-1, keepdim=True) / sub_probs.sum(dim=-1, keepdim=True)
        # mu.shape: (num_concurrent_solutions, )

        w = sub_probs * (sub_score - mu)
        # w.shape: (num_concurrent_solutions, )

        scale = w.abs().sum(dim=-1, keepdim=True)  
        advantage = torch.where(scale > 0, w / scale, w)
        # advantage.shape: (num_concurrent_solutions, )

        return advantage

                                
    def _loss_function(self, logits, ref_sel_log_probs, old_sel_log_probs, seq_ids, update_pos, advantage):
        # logits.shape: (batch, sequence, vocab)
        # ref_sel_log_probs.shape: (batch, sequence-1)
        # seq_ids.shape: (batch, sequence)
        # update_pos.shape: (batch, )
        # advantage.shape: (batch, )
    
        assert torch.all(update_pos >= 0)
    
        epsilon_up = self.optimizer_params['ppo_epsilon_up']
        epsilon_dn = self.optimizer_params['ppo_epsilon_dn']
        beta = self.optimizer_params['kld_beta']
        
        vocab_size = logits.size(2)
        logits_valid = torch.gather(logits, 1, update_pos[:, None, None].expand(-1, -1, vocab_size)).squeeze(1)
        # shape: (batch, vocab)
        
        seq_ids_shift = torch.gather(seq_ids[:, 1:], 1, update_pos[:, None]).squeeze(1)
        # shape: (batch, )
    
        log_probs = F.log_softmax(logits_valid, dim=1)
        # shape: (batch, vocab)
        
        sel_log_probs = torch.gather(log_probs, 1, seq_ids_shift[:, None]).squeeze(1)
        # shape: (batch, )
    
        if old_sel_log_probs == None:
            old_sel_log_probs = sel_log_probs.detach()
            # shape: (batch, )
    
        # Compute the surrogate losses:
        ratio = torch.exp(sel_log_probs-old_sel_log_probs)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - epsilon_dn, 1 + epsilon_up) * advantage
    
        # Compute KLD
        KLD = torch.exp(ref_sel_log_probs-sel_log_probs)-(ref_sel_log_probs-sel_log_probs)-1
        
        objective = torch.min(surrogate1, surrogate2)-beta*KLD
        # shape: (batch, )
    
        objective = objective.mean()
        loss = -objective
    
        return loss, old_sel_log_probs
    

