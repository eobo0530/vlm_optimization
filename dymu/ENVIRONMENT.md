# DToMe LLaVA 개발 환경 (Working Environment)

이 환경에서 DToMe가 적용된 LLaVA 벤치마크가 정상 동작합니다.

## 핵심 패키지 버전

```
torch==2.9.1
torchvision==0.24.1
transformers==4.57.3
tokenizers==0.22.2
numpy==1.26.4
timm==1.0.24
accelerate==1.12.0
safetensors==0.7.0
huggingface-hub==0.36.0
```

## Python 버전

```
python 3.10
```

## Conda 환경

```bash
conda activate dymu
```

## 환경 복구 방법

만약 환경이 꼬였다면:

```bash
# pymu 환경 활성화
conda activate dymu

# 핵심 패키지 재설치
pip install torch==2.9.1 torchvision==0.24.1
pip install transformers==4.57.3 tokenizers==0.22.2
pip install numpy==1.26.4 timm==1.0.24
```

## 벤치마크 실행 명령어

```bash
cd /home/user/vlm_opt_linux/dymu

PYTHONPATH=LLaVA:src:$PYTHONPATH python run_benchmark_dymu.py \
  --model-path checkpoints/vlm_checkpoints/llava-v1.5-7b \
  --data-file /home/user/vlm_opt_linux/microbench/perf_data/coco_val_500.json \
  --max-new-tokens 64 \
  --report-file report_coco_dymu.json
```

## 날짜

2026-01-11
