# VLMEvalKit Setup for Qwen2-VL (RTX 4090 / WSL2)

RTX 4090 (24GB) 및 WSL2 환경에서 `Qwen2-VL-7B` 모델을 `VLMEvalKit`으로 평가하기 위한 설치 및 실행 가이드입니다.

## 1. Environment Setup (환경 설정)

**System Requirements:**
- OS: Windows 11 (WSL2 Ubuntu)
- GPU: NVIDIA RTX 4090 (24GB)
- Driver: CUDA 12.8 supported driver

**Conda Environment:**
```bash
# 가상환경 생성 (Python 3.10 권장)
conda create -n vlmeval python=3.10 -y
conda activate vlmeval
2. Installation (설치)
2.1 Clone Repository
Bash

git clone [https://github.com/open-compass/VLMEvalKit.git](https://github.com/open-compass/VLMEvalKit.git)
cd VLMEvalKit
pip install -e .
2.2 Install Dependencies (Crucial)
Qwen2-VL 최신 모델 지원을 위해 transformers를 개발자 버전으로 설치해야 하며, 관련 유틸리티를 추가합니다.

Bash

# Install PyTorch (Compatible with CUDA 12.8)
pip install torch torchvision torchaudio

# Install bleeding-edge Transformers (Required for Qwen2-VL)
pip install git+[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

# Install Qwen utilities
pip install qwen-vl-utils
2.3 Optimization Strategy (Flash Attention vs SDPA)
Note: RTX 4090 환경에서 flash-attn 컴파일 시 시스템 RAM 부족(OOM) 및 CUDA 버전 미스매치 이슈가 발생할 수 있습니다. 따라서 **PyTorch Native SDPA(Scaled Dot-Product Attention)**를 사용하여 설치 없이 가속을 적용합니다.

설정 방법: flash-attn 설치를 건너뛰고, 실행 시 자동으로 SDPA가 적용되도록 하거나 코드에서 명시적으로 설정합니다. (만약 실행 시 FlashAttention2 에러가 발생한다면, 소스 코드의 모델 로드 부분에서 attn_implementation="flash_attention_2"를 제거하거나 "sdpa"로 변경합니다.)

3. Execution (실행)
MMBench (Dev) 평가 실행: 단일 GPU(4090)를 사용하여 평가를 수행합니다.

Bash

# MMBench_DEV_EN 데이터셋 평가
torchrun --nproc_per_node=1 run.py \
  --data MMBench_DEV_EN \
  --model Qwen2-VL-7B-Instruct \
  --verbose
  
4. Troubleshooting History
Issue 1: flash-attn 설치 실패 (RAM 부족으로 인한 컴파일러 강제 종료)

Solution: 설치를 포기하고 PyTorch 내장 sdpa 최적화 사용. 성능 차이는 미미함.