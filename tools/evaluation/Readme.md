
## 평가 및 response 생성에 lm-evaluation-harness 사용 (v0.4.8)
- https://github.com/EleutherAI/lm-evaluation-harness
- 직접 구현한 tasks 의 경우 tools/evaluation/lm_eval/tasks 아래 폴더를 lm-evaluation-harness/lm_eval/tasks 에 복사하여 사용


### datasets
- arc: https://huggingface.co/datasets/allenai/ai2_arc
- bbh: https://huggingface.co/datasets/lukaemon/bbh
- gpqa: https://huggingface.co/datasets/Idavidrein/gpqa
- gsm8k_platinum: https://huggingface.co/datasets/madrylab/gsm8k-platinum
- lcb: https://github.com/Cohere-Labs-Community/language-confusion
- mif: https://huggingface.co/datasets/AIDC-AI/Marco-Bench-MIF
- mmmlu: https://huggingface.co/datasets/openai/MMMLU
- math: https://huggingface.co/datasets/EleutherAI/hendrycks_math


## language list
- ko, zh, ar, ja


## locale_map
{
    "ko": "KO_KR",
    "zh": "ZH_CN",
    "ar": "AR_XY",
    "ja": "JA_JP",
}


## Tasks
- arc_challenge_chat
- bbh_cot_zeroshot
- gpqa_main_cot_zeroshot
- gpqa_diamond_cot_zeroshot
- gsm8k_platinum_cot_zeroshot
- mif_en
- score_non_greedy_robustness_math
- gsm8k_platinum_mix_{language}
- lcb_crosslingual_{language}
- lcb_monolingual_{language}
- mif_{language}
- mmmlu_{locale_map[language]}


## 직접 구현한 Tasks
- gsm8k_platinum_mix: gsm8k-platinum-cot-llama task 의 prompt 를 각 언어별 번역하여 사용
- mmmlu: simple-evals 참고하여 구현 (https://github.com/openai/simple-evals)
- mif: ifeval task 참고하여 구현
- lcb: 단순 llm generate


## TLPO 평가
- lm-evaluation-harness 실행 시 --log_samples 사용해 samples 생성하여 LLM response 사용
- lm-evaluation-harness output dir 을 TLPO_eval.py 의 --harness_output_dir 에 사용
- 영어는 confusion 예외 처리 한다면 --ignore_english 는 True
- confusion target_language 로 ko, zh, ar, ja
- output_dir 은 TLPO 결과 저장 위치

```
harness_output_dir=harness_output
output_dir=tlpo_output

python TLPO_eval.py \
    --harness_output_dir ${harness_output_dir} \
    --ignore_english True \
    --target_language "ko" \
    --output_dir ${output_dir}
```