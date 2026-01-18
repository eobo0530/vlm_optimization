# 하이브리드 VLM (DyMU + FastV) 배포 및 설치 가이드

본 프로젝트는 DyMU(Token Merging)와 FastV(Pruning)가 통합된 LLaVA 모델의 최적화된 구현체를 포함하고 있습니다. 복잡한 로컬 패치와 의존성을 자동으로 설치하기 위해 다음 절차를 따르십시오.

## 1. 환경 준비
새로운 Conda 환경을 생성하고 활성화합니다.

```bash
conda create -n vlm_hybrid_dist python=3.10 -y
conda activate vlm_hybrid_dist
```

## 2. 자동 설치 절차
프로젝트 루트 디렉토리(`vlm_optimization`)에서 설치 스크립트를 실행합니다.

```bash
# 스크립트 실행 권한 부여
chmod +x setup_hybrid.sh

# 전체 설치 수행 (transformers, LLaVA, dymu, VLMEvalKit 포함)
./setup_hybrid.sh
```

## 3. 구성 요소 상세
설치 스크립트는 다음 구성 요소들을 에디터블 모드(`pip install -e`)로 설치합니다:
- **FastV/src/transformers**: FastV 패치가 적용된 v4.31.0
- **FastV/src/LLaVA**: DyMU 패치가 적용된 LLaVA 구현체
- **dymu**: 토큰 머징 핵심 라이브러리
- **VLMEvalKit**: 벤치마크 수행 툴킷

## 4. 실행 예시
설치 완료 후 다음 명령어로 벤치마크를 수행할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python VLMEvalKit/run.py --data COCO_VAL --model llava_v1.5_7b_hybrid --verbose
```
