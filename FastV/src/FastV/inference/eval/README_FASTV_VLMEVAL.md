# FastV + VLMEvalKit 통합 가이드

이 디렉토리에는 FastV를 VLMEvalKit과 통합하여 MMBench 및 COCO 데이터셋에서 평가하는 스크립트가 포함되어 있습니다.

## 파일 설명

### 핵심 파일

1. **`fastv_wrapper.py`**: FastV를 지원하는 커스텀 LLaVA 모델 클래스
   - VLMEvalKit의 `BaseModel`을 상속
   - LLaVA 모델 로드 후 FastV 설정을 주입
   - `generate_inner()` 메서드를 통해 VLMEvalKit과 통합

2. **`run_fastv_vlmeval.py`**: 메인 평가 스크립트
   - VLMEvalKit의 evaluation framework 사용
   - MMBench, COCO 등 다양한 데이터셋 지원
   - FastV 파라미터를 커맨드라인 인자로 설정 가능

3. **`config_fastv_example.json`**: 예제 설정 파일
   - JSON 형식으로 모든 설정을 관리
   - 재현 가능한 실험을 위한 설정 저장

4. **`run_fastv_vlmeval.sh`**: 실행 예제 쉘 스크립트
   - 빠른 실행을 위한 샘플 스크립트

## 사용 방법

### 방법 1: 커맨드라인 인자 사용

```bash
python run_fastv_vlmeval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-fast-v \
    --fast-v-attention-rank 100 \
    --data MMBench_DEV_EN_V11 COCO_VAL \
    --work-dir ./outputs_fastv \
    --verbose
```

### 방법 2: JSON 설정 파일 사용

```bash
python run_fastv_vlmeval.py --config config_fastv_example.json
```

### 방법 3: 쉘 스크립트 사용

```bash
bash run_fastv_vlmeval.sh
```

## FastV 파라미터 설명

- `--use-fast-v`: FastV 토큰 가지치기 활성화
- `--fast-v-sys-length` (기본값: 35): 시스템 프롬프트 토큰 길이
- `--fast-v-image-token-length` (기본값: 576): 이미지 토큰 개수 (24x24 패치)
- `--fast-v-attention-rank` (기본값: 100): 유지할 상위 attention 토큰 개수
  - **이 값을 조정하여 프루닝 비율을 제어합니다**
  - 값이 작을수록 더 많은 토큰이 제거됩니다
- `--fast-v-agg-layer` (기본값: 3): attention 점수를 집계할 레이어 인덱스

## 지원 데이터셋

VLMEvalKit에서 지원하는 모든 데이터셋을 사용할 수 있습니다. 주요 데이터셋:

- **MMBench 시리즈**: `MMBench_DEV_EN_V11`, `MMBench_TEST_EN_V11`, `MMBench_CN_V11`
- **COCO**: `COCO_VAL` (COCO Captioning validation set)
- **기타**: `POPE`, `GQA`, `TextVQA`, `VizWiz`, `ScienceQA` 등

전체 데이터셋 목록 확인:
```python
from vlmeval.dataset import SUPPORTED_DATASETS
print(SUPPORTED_DATASETS)
```

## 출력 구조

```
outputs_fastv/
├── llava-1.5-7b-fastv_rank100_MMBench_DEV_EN_V11/
│   ├── llava-1.5-7b-fastv_rank100_MMBench_DEV_EN_V11.xlsx
│   └── llava-1.5-7b-fastv_rank100_MMBench_DEV_EN_V11_results.json
└── llava-1.5-7b-fastv_rank100_COCO_VAL/
    ├── llava-1.5-7b-fastv_rank100_COCO_VAL.xlsx
    └── llava-1.5-7b-fastv_rank100_COCO_VAL_results.json
```

## 주의사항

1. **원본 LLaVA 형식 모델 사용**: `liuhaotian/llava-v1.5-7b` (또는 13b)
   - ❌ `llava-hf/llava-1.5-7b-hf` (Hugging Face 형식은 지원 안 됨)
   - ✅ `liuhaotian/llava-v1.5-7b` (원본 LLaVA 형식)
2. VLMEvalKit이 `/home/aips/VLMEvalKit`에 설치되어 있어야 합니다
3. LLaVA 라이브러리가 설치되어 있어야 합니다 (`src/LLaVA`)
4. FastV 구현이 모델에 포함되어 있어야 합니다 (`reset_fastv()` 메서드 필요)

## 사용 예제

### 기본 실행 (원본 LLaVA 7B 모델)

```bash
cd /home/aips/FastV/src/FastV/inference/eval
python run_fastv_vlmeval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-fast-v \
    --fast-v-attention-rank 100 \
    --data MMBench_DEV_EN_V11 COCO_VAL \
    --verbose
```

### 13B 모델 사용

```bash
python run_fastv_vlmeval.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --use-fast-v \
    --fast-v-attention-rank 100 \
    --data MMBench_DEV_EN_V11 \
    --verbose
```

### 다양한 pruning ratio 실험

```bash
# 더 공격적인 pruning (rank=50)
python run_fastv_vlmeval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-fast-v \
    --fast-v-attention-rank 50 \
    --data MMBench_DEV_EN_V11 \
    --verbose

# 보수적인 pruning (rank=200)
python run_fastv_vlmeval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-fast-v \
    --fast-v-attention-rank 200 \
    --data MMBench_DEV_EN_V11 \
    --verbose
```

## 다른 평가 파일과의 차이점

- **`inference_aokvqa.py`**, **`inference_ocrvqa.py`**: 
  - 특정 데이터셋에 특화된 standalone 스크립트
  - 직접 데이터를 로드하고 평가 로직을 구현
  
- **`run_fastv_vlmeval.py`** (이 파일):
  - VLMEvalKit의 통합 프레임워크 사용
  - 여러 데이터셋을 단일 스크립트로 평가
  - 표준화된 평가 파이프라인 활용
