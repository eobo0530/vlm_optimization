# DToMe LLaVA 1.5 통합 요점 및 가이드

이 문서는 DToMe(Dynamic Token Merging)를 LLaVA 1.5 모델에 통합하는 과정에서 발생한 모든 문제와 해결책, 그리고 벤치마크 실행 방법을 기록한 종합 가이드입니다.

---

## 1. 주요 변경 사항 및 문제 해결 (History of Fixes)

LLaVA(레거시 패키지), timm, transformers, torch 간의 복잡한 버전 충돌 및 호환성 문제를 해결하기 위해 다음과 같은 조치를 취했습니다.

### A. 모델 설정 및 가중치 (Configuration)
*   **Threshold 계산 생략**: 직접 Threshold를 학습/계산하는 과정이 매우 복잡하고 시간이 오래 걸리므로, 미리 계산된 체크포인트(`ViT-L-14-336-tome-72out.pth`)를 Hugging Face에서 다운로드하여 사용하여 시간을 단축했습니다.
*   **config.json 수정**: `checkpoints/vlm_checkpoints/llava-v1.5-7b/config.json` 파일의 `mm_vision_tower` 경로를 변경하여 DToMe가 적용된 Vision Transformer를 사용하도록 설정했습니다.
*   **.gitignore 수정**: `checkpoints/` 폴더 내의 파일 수정이 무시되지 않도록 설정을 변경했습니다.

### B. 코드베이스 호환성 패치 (Codebase Patches)
최신 라이브러리 환경(`torch 2.9.1`, `transformers 4.57.3`)에서 구형 Codebase인 LLaVA가 동작하도록 수많은 패치를 적용했습니다.

1.  **`torch-scatter` 제거 및 대체**:
    *   **문제**: `torch-scatter` 설치 문제 발생.
    *   **해결**: `LLaVA/llava/model/language_model/llava_llama.py` 내의 `scatter_mean` 함수를 외부 라이브러리 없이 순수 PyTorch로 직접 구현하여 대체했습니다.

2.  **Transformers 4.57.3 호환성 확보**:
    *   **문제**: `transformers` 업데이트로 인해 `Unpack`, `FlashAttentionKwargs`, `_update_causal_mask` 등이 제거되거나 변경되어 오류 발생.
    *   **해결**: 
        *   `llava_llama.py`에서 삭제된 `_update_causal_mask` 메서드 호출을 최신 `create_causal_mask` 함수(transformers.models.llama.modeling_llama)로 교체했습니다.
        *   호환되지 않는 import 구문(`Unpack`, `FlashAttentionKwargs`)을 정리했습니다.

3.  **DToMe 텐서 차원 불일치 해결 (RuntimeError)**:
    *   **문제**: Token Merging 과정에서 텐서의 크기가 변경되면서, 복원(Unmerging) 시 `scatter_add_` 연산에서 Index 텐서와 Source 텐서의 차원이 맞지 않는 오류(`RuntimeError`) 발생.
    *   **해결**: `llava_llama.py`의 `remerge_mapping_hidden_states` (및 내부 `scatter_mean`) 함수에 브로드캐스팅 로직을 추가하여, 1D Index 텐서가 3D/4D Source 텐서와 올바르게 연산되도록 차원을 자동으로 확장(unsqueeze/expand)하게 수정했습니다.

4.  **`safetensors` 로딩 방지**:
    *   **문제**: `.bin` 파일만 있는 체크포인트를 `safetensors`로 로드하려다 실패.
    *   **해결**: `builder.py`에서 강제로 `use_safetensors=False` 옵션을 추가했습니다.

---

## 2. DToMe 벤치마크 실행 방법

DToMe가 적용된 LLaVA 모델의 성능을 측정하기 위해 전용 스크립트(`run_benchmark_dymu.py`)를 제작했습니다. 이 스크립트는 `microbench/benchmark_perf.py`의 로직을 기반으로 하되, DToMe 설정을 주입하도록 개조되었습니다.

### 사전 준비 (Environment)
```bash
# dymu 디렉토리로 이동
cd /home/user/vlm_opt_linux/dymu

# PYTHONPATH 설정 (필수)
export PYTHONPATH=src:$PYTHONPATH
```

### 실행 명령어

**1. 추론 테스트 (Inference Test)**
모델이 정상적으로 문장을 생성하는지 확인합니다.
```bash
python LLaVA/inference_dymu_llava.py
```

**1. COCO Val 500 (Subset) 측정**
가장 일반적인 성능 측정용 시나리오입니다.
```bash
python run_benchmark_dymu.py \
  --model-path checkpoints/vlm_checkpoints/llava-v1.5-7b \
  --data-file /home/user/vlm_opt_linux/microbench/perf_data/coco_val_500.json \
  --max-new-tokens 64 \
  --report-file report_coco_dymu.json
```

**2. MMBench DEV EN (Full) 측정**
전체 데이터셋에 대한 측정이 필요할 때 사용합니다.
```bash
python run_benchmark_dymu.py \
  --model-path checkpoints/vlm_checkpoints/llava-v1.5-7b \
  --data-file /home/user/vlm_opt_linux/microbench/perf_data/mmbench_dev_en.json \
  --max-new-tokens 128 \
  --report-file report_mmbench_dymu.json
```

---

## 3. 관련 파일 위치

*   **벤치마크 스크립트:** `/home/user/vlm_opt_linux/dymu/run_benchmark_dymu.py`
*   **수정된 모델 코드:** `/home/user/vlm_opt_linux/dymu/LLaVA/llava/model/language_model/llava_llama.py`
*   **Threshold 체크포인트:** `/home/user/vlm_opt_linux/dymu/checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth`
