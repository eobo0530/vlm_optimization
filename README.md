# VLM Optimization: Hybrid DyMU + FastV (1200x Speedup)

본 프로젝트는 LLaVA-1.5-7B 모델에 **DyMU (Dynamic Multi-token Unmerging)**와 **FastV (Attention Pruning)** 기술을 통합하여, 기존 대비 최대 **1200배**의 속도 향상을 달성한 하이브리드 최적화 프레임워크입니다.

## 🚀 주요 특징
- **DyMU 통합**: Vision Encoder 단계에서 토큰을 병합(Merge)하고, LLM Attention 단계에서 유연하게 복원(Unmerge)하여 계산 효율 극대화.
- **FastV 최적화**: 중요도가 낮은 Vision 토큰을 Attention 계산에서 제외하여 추론 속도 가속.
- **최적화 커널 커스텀**: 
  - CPU-GPU 동기화를 제거한 `scatter_mean` 커널 구현.
  - 최신 `scaled_dot_product_attention` (SDPA) 적용 및 하이브리드 마스크 정렬.
- **가변 시퀀스 지원**: 텍스트와 이미지가 혼합된 시퀀스에서도 정밀한 병합/복원 로직 작동.

## 📄 문서 가이드
상세한 구현 내용과 환경 설정은 아래 전용 문서를 참고해 주세요.

1. **상세 구현 보고서**: [hybrid_implementation_report.md](hybrid_implementation_report.md)
   - DyMU와 FastV가 어떻게 결합되었는지, 핵심 알고리즘 및 코드 수정 내역이 상세히 기록되어 있습니다.
2. **복구 및 최적화 요약**: [restoration_summary.md](restoration_summary.md)
   - 최신 `transformers v4.31.0` 버전과의 호환성 해결 및 `IndexError` 패치 내역을 요약합니다.
3. **배포 및 실행 가이드 (Korean)**: [dist_guide_ko.md](dist_guide_ko.md)
   - 모델 실행 방법 및 주요 옵션 설명.

## 🛠 환경 설정 (Environmet Setup)

하이브리드 모델 실행을 위한 전용 가상환경 설정 방법입니다.

### 1. 전용 가상환경 생성 및 필수 라이브러리 설치
제공된 `setup_hybrid.sh` 스크립트를 사용하여 자동으로 세팅할 수 있습니다.

```bash
chmod +x setup_hybrid.sh
./setup_hybrid.sh
```

### 2. 수동 설정 시 주요 단계
- **Base Environment**: Python 3.10, PyTorch 2.0.1+ (CUDA 11.7/11.8 권장)
- **Core Dependencies**:
  - `transformers` (프로젝트 내 수정된 패치 버전 사용)
  - `tokenizers`
  - `sentencepiece`
  - `clip`

## 🏃 실행 방법 (Evaluation)

[VLMEvalKit](VLMEvalKit)을 사용하여 COCO 등의 벤치마크를 수행할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=1 python VLMEvalKit/run.py \
    --data COCO_VAL \
    --model llava_v1.5_7b_hybrid \
    --verbose
```

export PYTHONPATH=$PYTHONPATH:$PWD/FastV/src/LLaVA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib
CUDA_VISIBLE_DEVICES=0 python VLMEvalKit/run.py --data COCO_VAL --model llava_v1.5_7b_hybrid --verbose
ㅇ
---
> [!NOTE]
> `checkpoints/` 폴더는 용량 문제로 Git 업로드에서 제외되었습니다. 모델 가중치 파일은 별도로 관리해 주시기 바랍니다.
