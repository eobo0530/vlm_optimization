# VLMEvalKit Setup for LLaVA (RTX 4090 / WSL2)

RTX 4090 (24GB) 및 WSL2 환경에서 `LLaVA` (LLaVA-v1.5-7b 등) 모델을 `VLMEvalKit`으로 평가하기 위한 설치 및 실행 가이드입니다.

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
```

## 2. Installation (설치)

### 2.1 Clone Repository

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

### 2.2 Install Dependencies (Crucial for LLaVA)

LLaVA 모델 실행을 위해 LLaVA 패키지와 관련 의존성을 설치해야 합니다.

```bash
# Install PyTorch (Compatible with CUDA 12.8)
pip install torch torchvision torchaudio

# Install LLaVA
pip install --upgrade pip
pip install "git+https://github.com/haotian-liu/LLaVA.git"

# Install COCO evaluation tools (Optional, for COCO Caption)
pip install pycocoevalcap

# Note: Bleeding-edge transformers are NOT compatible with LLaVA 1.5
# pip install git+https://github.com/huggingface/transformers
```

### 2.3 Optimization Strategy (Flash Attention vs SDPA)

Note: RTX 4090 환경에서 `flash-attn` 컴파일 시 시스템 RAM 부족(OOM) 및 CUDA 버전 미스매치 이슈가 발생할 수 있습니다. 따라서 **PyTorch Native SDPA(Scaled Dot-Product Attention)**를 사용하여 설치 없이 가속을 적용하는 것이 안정적일 수 있습니다.

만약 `flash-attn`을 사용하려면 별도로 설치해야 하며, 설치 실패 시 기본적으로 SDPA가 사용되거나 HF transformers 구현체가 사용됩니다.

## 3. Execution (실행)

MMBench (Dev) 평가 실행: 단일 GPU(4090)를 사용하여 평가를 수행합니다.
LLaVA v1.5 7B 모델을 기준으로 합니다.

```bash
# MMBench_DEV_EN 데이터셋 평가
PYTHONPATH=../LLaVA:$PYTHONPATH torchrun --nproc_per_node=1 run.py \
  --data MMBench_DEV_EN \
  --model llava_v1.5_7b \
  --verbose

# COCO 데이터셋 (Image Captioning) 평가
PYTHONPATH=../LLaVA:$PYTHONPATH torchrun --nproc_per_node=1 run.py \
  --data COCO_VAL \
  --model llava_v1.5_7b \
  --verbose
```

## 4. Troubleshooting History

**Issue 1: flash-attn 설치 실패**
- 증상: RAM 부족으로 인한 컴파일러 강제 종료 등.
- 해결: 설치를 건너뛰고 PyTorch 내장 최적화(SDPA)를 사용하거나, 메모리가 충분한 환경에서 빌드된 wheel 파일을 사용.