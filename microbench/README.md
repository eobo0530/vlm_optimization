# VLM Performance Benchmark (Microbench)

이 디렉토리는 VLM(Vision Language Model)의 **Inference 성능 측정**을 위한 독립적인 벤치마크 도구들을 포함하고 있습니다.
`VLMEvalKit`에 의존성을 가지지만, 성능 측정에 불필요한 복잡성을 제거하고 **통제된 환경(Fixed Inputs/Configs)**에서 정밀하게 지표를 측정하는 것을 목표로 합니다.

## 1. 주요 파일 구성

- `prepare_perf_data.py`: 성능 측정용 고정 데이터셋(JSON) 생성 스크립트
- `benchmark_perf.py`: 실제 모델 로딩 및 인퍼런스 수행, 성능 지표(TTFT, TPS, VRAM 등) 측정 스크립트
- `perf_data/`: 생성된 데이터셋이 저장되는 폴더
  - `coco_val_500.json`: COCO Validation Set 중 고정된 500개 샘플
  - `mmbench_dev_en.json`: MMBench DEV (English) 전체 데이터셋

---

## 2. 데이터셋 준비 (Dataset Preparation)

성능 비교의 공정성을 위해 **입력 데이터와 순서를 고정**합니다.
`VLMEvalKit`를 통해 원본 데이터를 다운로드/로딩한 후, 필요한 정보(이미지 경로, 프롬프트)만 추출하여 JSON으로 저장합니다.

### 특징
- **Seed 고정**: `seed=42`를 사용하여 언제나 동일한 샘플이 선택됩니다.
- **실행 순서 고정**: 인덱스(Index) 기준으로 정렬(Sort)하여 저장하므로, 캐시(Cache)나 워밍업(Warm-up) 효과가 매번 동일하게 작용합니다.
- **경로 정규화**: 이미지 경로를 상대 경로로 변환하여 저장합니다.

### 실행 방법
```bash
python prepare_perf_data.py
```
위 명령어를 실행하면 `perf_data/` 폴더 내의 JSON 파일들이 갱신됩니다.

---

## 3. 벤치마크 실행 (Benchmark Execution)

준비된 데이터셋을 로드하여 모델의 추론 성능을 측정합니다.

### 측정 지표
- **E2E Latency**: End-to-End 전체 소요 시간 (초)
- **TTFT (Time To First Token)**: 첫 토큰 생성까지 걸린 시간 (Prefill 속도)
- **TBT (Time Between Tokens)**: 토큰 간 생성 시간 (Decode Latency, 초)
- **TPS (Tokens Per Second)**: 디코딩 단계의 초당 토큰 생성 속도 (Decode 속도 = 1/TBT)
- **Peak VRAM**: GPU 메모리 최대 사용량 (MB)

### 고정된 설정 (Fixed Configs)
- **Decoding**: Greedy (sample=False), `temperature=0`
- **Max New Tokens**: 사용자 지정 (기본 512)

### 실행 예시

**1. COCO Val 500 (Subset) 측정**
가장 일반적인 성능 측정용 시나리오입니다.
```bash
python benchmark_perf.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --data-file perf_data/coco_val_500.json \
  --max-new-tokens 64 \
  --report-file report_coco_baseline.json
```

**2. MMBench DEV EN (Full) 측정**
전체 데이터셋에 대한 측정이 필요할 때 사용합니다.
```bash
python benchmark_perf.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --data-file perf_data/mmbench_dev_en.json \
  --max-new-tokens 128 \
  --report-file report_mmbench_baseline.json
```

---

## 4. 참고 사항
- 스크립트는 `microbench` 폴더 내에서 실행하는 것을 권장합니다.
- 이미지 경로는 `../VLMEvalKit` 등 상위 디렉토리의 자원을 참조할 수 있습니다.
- `benchmark_perf.py` 실행 시 `transformers`나 `llava` 패키지가 설치된 환경(conda env)이어야 합니다.
