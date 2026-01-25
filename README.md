# VLM Optimization: Hybrid DyMU + FastV (Ultra-Fast Inference)

본 프로젝트는 LLaVA-1.5-7B 모델에 **DyMU (Dynamic Multi-token Unmerging)**와 **FastV (Attention Pruning)** 기술을 최적화하여 구현한 하이브리드 VLM 프레임워크입니다. 

기존의 고해상도 이미지 처리 병목을 해결하기 위해 **Logical RoPE Mapping**과 **Conditional SDPA** 기술을 적용하여, 시각적 정보의 손실 없이 비약적인 추론 속도 향상을 달성했습니다.

---

## 🚀 핵심 기술 (Key Logic)

### 1. Logical RoPE Mapping (DyMU 개선)
병합(Merge)된 시각 토큰들이 LLM 내에서도 정확한 위치 정보를 유지할 수 있도록 **논리적 위치 기반 RoPE(Rotary Positional Embedding)**를 적용했습니다.
- **원리**: 576개의 원본 토큰 위치 중 병합 후 남은 대표 토큰의 위치 인덱스를 추적하여, 해당 토큰에 원래의 Position ID를 할당합니다.
- **효과**: 토큰 수가 줄어들어도 모델이 이미지의 공간적 구조를 정확히 인식하여 성능 하락을 방지합니다.

### 2. FastV Attention Pruning
이미지 토큰 중 질문(Query)과의 연관성이 낮은 토큰을 연산에서 제외하여 가속화합니다.
- **K-Rank Pruning**: Aggregation Layer에서 계산된 어텐션 가중치를 바탕으로 상위 K개의 중요한 토큰만 남기고 나머지는 마스킹 처리합니다.

### 3. Conditional SDPA 및 Zero-Sync 최적화
- **Conditional SDPA**: FastV가 어텐션 가중치를 필요로 하는 특정 레이어(Layer 2)를 제외한 나머지 모든 레이어에서 하드웨어 가속인 `scaled_dot_product_attention`을 사용합니다.
- **Zero-Sync 커널**: CPU-GPU 동기화를 유발하는 `.item()` 호출 등을 제거하고, 전 과정을 GPU 내에서 비동기 연산으로 처리하도록 개선했습니다.

---

## 🛠 환경 설정 (Setup)

### 가상환경 준비
`vlm_hybrid` 전용 conda 환경을 권장합니다.

```bash
# 가상환경 활성화
conda activate vlm_hybrid
```

### 환경 변수 설정
실행 전 다음 환경 변수들을 설정하여 인터페이스를 정렬합니다.
```bash
# 소스 경로 주입
export PYTHONPATH="/home/aips/vlm/dymu/src:/home/aips/vlm/FastV/src/transformers/src:/home/aips/vlm/FastV/src/LLaVA:${PYTHONPATH}"

# FastV K값 및 동적 비율(Ratio) 설정
- **정적 토큰 수**: `FASTV_K=72`와 같이 1 이상의 정수를 입력하면 고정된 개수의 토큰만 남깁니다.
- **동적 프루닝 비율**: `FASTV_K=0.5`와 같이 0에서 1 사이의 소수를 입력하면 비율 기반으로 프루닝합니다.
    - 예: `0.7` 입력 시 전체 시각 토큰 중 **70%를 제거(Pruning)**하고 30%만 유지합니다.
    - DyMU에 의해 이미지마다 토큰 수가 달라지는 경우(예: 105개, 165개 등)에도 일정한 비율로 최적화가 가능합니다.

---

## 🏃 벤치마크 실행 방법 (Execution)

[VLMEvalKit](VLMEvalKit)을 사용하여 모델의 성능과 속도를 정밀하게 측정할 수 있습니다.

### MMBench 실행 예시 (K=36)
```bash
CUDA_VISIBLE_DEVICES=0 /home/aips/miniconda3/envs/vlm_hybrid/bin/python run.py \
    --data MMBench_DEV_EN \
    --model llava_v1.5_7b_hybrid \
    --verbose \
    --work-dir hybrid_mmbench_k36 \
    --reuse
```

### COCO Captioning 실행 예시 (K=72)
```bash
export FASTV_K=72
CUDA_VISIBLE_DEVICES=1 /home/aips/miniconda3/envs/vlm_hybrid/bin/python run.py \
    --data COCO_VAL \
    --model llava_v1.5_7b_hybrid \
    --verbose \
    --work-dir hybrid_coco_k72
```

---

## 📊 현재 성과 및 지표
- **정확도**: `MMBench_DEV_EN` 기준 **0.553 (Overall)** 확보.
- **추론 속도**: 최적화 후 이미지당 처리 속도 대폭 개선 (성능 보고서 참고).

## 📄 추가 문서
- [상세 구현 보고서](hybrid_implementation_report.md): 알고리즘 융합 및 커널 수정 내역
- [복구 요약서](restoration_summary.md): Transformers v4.31.0 호환성 및 버그 수정 내역

---
> [!IMPORTANT]
> 실행 시 반드시 `PYTHONPATH`에 통합된 소스 코드가 포함되어 있는지 확인하십시오. 패키지 형태가 아닌 소스 주입 방식으로 동작합니다.
