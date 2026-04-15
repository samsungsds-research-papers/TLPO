
## Bactrian-X 데이터셋을 이용해 학습 데이터 생성
- https://huggingface.co/datasets/MBZUAI/Bactrian-X
- SFT 는 Bactrian-X 그대로 사용


### 1. train data filter
- 언어, 개발, 터미널 관련 단어 필터링
- question 추출

```
dataset_dir=dataset/Bactrian-X
output_dir=dataset/Bactrian-X-filtered/data
target_language=ko

python train_data_filter.py \
    --dataset_dir=${dataset_dir} \
    --output_dir=${output_dir} \
    --target_language=${target_language}
```


### 2. make bactrian sample
- 1에서 생성한 question 이용해 각 모델 별 16개 답변 샘플링
- question/answers 형태로 생성
- {dataset_dir}/{model_type}/{target_language}.jsonl 에 저장

```
dataset_dir=dataset/Bactrian-X-filtered/data
model_type=hf
model_path=meta-llama/Llama-3.1-8B-Instruct
target_language=ja

python make_bactrian_sample.py \
    --model_type=${model_type} \
    --model_path=${model_path} \
    --target_language=${target_language}
```


### 3. make train dataset
- 2에서 생성한 16개 샘플 중 language confusion 발생 여부 체크하여 데이터 구분
- confusion 발생 answer 가 0개, 16개인 데이터는 제외
- prompt: question
- chosen: none confusion list 에서 1개 랜덤 추출
- rejected: confusion list 에서 1개 랜덤 추출
- prompt/chosen/rejected 데이터셋 구성하여 dpo/orpo 에 사용

```
dataset_dir=dataset/Bactrian-X-filtered/data
output_dir=train_data
target_language=zh
ignore_english=true

python make_train_dataset.py \
    --dataset_dir=${dataset_dir} \
    --output_dir=${output_dir} \
    --target_language=${target_language} \
    --ignore_english=${ignore_english}
```