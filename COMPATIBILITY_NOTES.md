# 환경 호환성 노트

## 현재 환경 vs 새 노트북 환경

### 현재 개발 환경 (RTX 3060 Ti)
- **GPU**: NVIDIA RTX 3060 Ti (Ampere, sm_86)
- **CUDA**: 12.1 (드라이버 CUDA 12.9 지원)
- **PyTorch**: 2.4.1+cu121
- **Python**: 3.11.6
- **상태**: 안정적, 모든 기능 정상 작동

### 새 노트북 환경 (RTX 5070)
- **GPU**: NVIDIA RTX 5070 (Blackwell, sm_120)
- **CUDA**: 12.8 이상 필수
- **PyTorch**: Nightly 빌드 필수 (stable 미지원)
- **Python**: 3.11.6 (동일)
- **상태**: 실험적, nightly 빌드 필요

## 호환성 이슈

### 주요 문제점
1. **아키텍처 차이**
   - RTX 3060 Ti: Ampere (sm_86)
   - RTX 5070: Blackwell (sm_120)
   - Blackwell은 새로운 compute capability 필요

2. **PyTorch 버전 충돌**
   - Stable PyTorch 2.4.1은 sm_120 미지원
   - RTX 5070은 PyTorch nightly 필수
   - Nightly 버전은 불안정할 수 있음

3. **CUDA 버전 차이**
   - RTX 3060 Ti: CUDA 12.1로 충분
   - RTX 5070: CUDA 12.8 최소 요구

## 해결 방안

### 옵션 1: 각 환경별 별도 설정 (권장)
```bash
# RTX 3060 Ti 환경
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# RTX 5070 환경
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 옵션 2: 환경 감지 코드 추가
```python
import torch
import warnings

def check_gpu_compatibility():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
        
        if capability[0] == 12 and capability[1] == 0:  # sm_120
            warnings.warn(f"Detected {device_name} with sm_120. "
                         "Make sure you're using PyTorch nightly with CUDA 12.8+")
        
        return capability, device_name
    return None, None

# 사용 예시
capability, device = check_gpu_compatibility()
if capability:
    print(f"GPU: {device}, Compute Capability: {capability[0]}.{capability[1]}")
```

### 옵션 3: Docker 컨테이너 사용
```dockerfile
# RTX 5070용 Dockerfile
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 코드 수정 권장사항

### 1. GPU 감지 및 설정
```python
import torch

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # RTX 5070 특화 설정
        if 'RTX 5070' in torch.cuda.get_device_name(0):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("RTX 5070 detected - TF32 enabled")
    else:
        device = torch.device('cpu')
    return device
```

### 2. Mixed Precision 활용
```python
from torch.cuda.amp import autocast, GradScaler

# 두 GPU 모두에서 성능 향상
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

## 테스트 체크리스트

새 노트북에서 테스트 시:
- [ ] PyTorch nightly 설치 확인
- [ ] CUDA 12.8 설치 확인
- [ ] GPU 인식 테스트
- [ ] 간단한 텐서 연산 테스트
- [ ] 모델 로드 테스트
- [ ] 전체 training 파이프라인 테스트
- [ ] 성능 벤치마크 (RTX 3060 Ti vs RTX 5070)

## 향후 계획

1. **단기 (1-2개월)**
   - PyTorch stable이 sm_120 지원할 때까지 nightly 사용
   - 코드 호환성 테스트 지속

2. **중기 (3-6개월)**
   - PyTorch stable 버전 업데이트 시 전환
   - 모든 환경 통일

3. **장기**
   - RTX 3060 Ti 환경 단계적 폐기
   - RTX 5070 최적화 코드로 전환

## 참고 링크

- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [PyTorch GPU Support Matrix](https://pytorch.org/get-started/locally/#linux-prerequisites)