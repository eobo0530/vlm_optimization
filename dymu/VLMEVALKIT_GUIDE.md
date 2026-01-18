# DToMe LLaVA VLMEvalKit 실행 가이드

DToMe가 적용된 LLaVA 모델로 VLMEvalKit 벤치마크를 실행하는 방법입니다.

---

## 사전 준비

```bash
# dymu_eval 환경 활성화
conda activate dymu_eval

# dymu 폴더로 이동
cd /home/user/vlm_opt_linux/dymu
```

---

## 벤치마크 실행 명령어

### 1. COCO_VAL (Caption 평가)

```bash
PYTHONPATH=LLaVA:src:$PYTHONPATH python ../VLMEvalKit/run.py \
  --data COCO_VAL \
  --model llava_v1.5_7b_dymu \
  --verbose
```

### 2. MMBench (다중 선택 평가)

```bash
PYTHONPATH=LLaVA:src:$PYTHONPATH python ../VLMEvalKit/run.py \
  --data MMBench_DEV_EN \
  --model llava_v1.5_7b_dymu \
  --verbose
```


## 결과 저장 위치

결과는 자동으로 `outputs/` 폴더에 저장됩니다:
- `outputs/llava_v1.5_7b_dymu/COCO_VAL/`
- `outputs/llava_v1.5_7b_dymu/MMBench_DEV_EN/`

---

## 주의사항

1. **반드시 `dymu/` 폴더에서 실행해야 합니다** (체크포인트 경로 문제)
2. `.env` 파일 경고는 무시해도 됩니다
3. `pynvml` 경고도 무시 가능

---

## 환경 정보

- Conda 환경: `dymu_eval`
- torch: 2.9.1
- transformers: 4.57.3
- numpy: 2.2.6

---

## 날짜

2026-01-11
