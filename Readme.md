# TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models

This repository is the official implementation of **TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models**(ACL 2026).

<p align="center">
<img src="./images/tlpo_ab.png" alt="tlpo_ab" width="45%">
<img src="./images/tlpo_c.png" alt="tlpo_c" width="45%">
</p>

paper url: TBD

<br>


#### 주요 폴더 및 파일 설명
- main.py: entry point of TLPO
- tools / fitler_data : 학습 data 준비
- tools / evaluation : 평가 관련 코드


#### Dataset 준비
train: Bactrian-X (https://huggingface.co/datasets/MBZUAI/Bactrian-X)
+ tools / filter_data에서 data 준비하는 방법
+ main.py에 dataset 위치 지정하는 방법


#### Model 준비
- meta-llama/Llama-3.1-8B-Instruct: (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Qwen/Qwen3-8B: (https://huggingface.co/Qwen/Qwen3-8B)
- google/gemma-3-4b-it: (https://huggingface.co/google/gemma-3-4b-it)
- mistralai/Ministral-8B-Instruct-2410: (https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)


#### Run TLPO
```
python main.py
```

#### Evaluation

+ lm-eval-harness 를 이용하여 Evaluation 수행
+ tools / evaluation 을 lm-eval-harness에 추가하라는 내용 추가

아래 내용 설명  
eval:
- arc: https://huggingface.co/datasets/allenai/ai2_arc
- bbh: https://huggingface.co/datasets/lukaemon/bbh
- gpqa: https://huggingface.co/datasets/Idavidrein/gpqa
- gsm8k_platinum: https://huggingface.co/datasets/madrylab/gsm8k-platinum
- lcb: https://github.com/Cohere-Labs-Community/language-confusion
- mif: https://huggingface.co/datasets/AIDC-AI/Marco-Bench-MIF
- mmmlu: https://huggingface.co/datasets/openai/MMMLU
- math: https://huggingface.co/datasets/EleutherAI/hendrycks_math


<br>
