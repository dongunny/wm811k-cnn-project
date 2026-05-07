# WM-811K 웨이퍼 맵 결함 패턴 분류 (CNN)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Balanced_Accuracy-92.02%25-brightgreen.svg)]()

> WM-811K 웨이퍼 맵 이미지를 활용하여 8가지 반도체 결함 패턴을 CNN으로 분류하는 프로젝트입니다.
> **Phase 2** of the Semiconductor Process AI Pipeline (반도체 공정 AI 파이프라인)

---

## 🎯 프로젝트 개요

반도체 제조 공정에서 웨이퍼 맵(Wafer Map)에 나타나는 결함 패턴은 어떤 공정 단계에서 문제가 발생했는지를 알려주는 핵심 정보입니다.
이 프로젝트는 **811,457장의 웨이퍼 맵 이미지**를 학습하여, 8가지 결함 유형을 자동으로 분류하는 **CNN(합성곱 신경망)** 모델을 구축합니다.

- **이전 프로젝트(Phase 1):** [secom-fdc-project](https://github.com/dongunny/secom-fdc-project) — SECOM 센서 데이터 → XGBoost + AutoEncoder로 불량 탐지
- **이번 프로젝트(Phase 2):** WM-811K 이미지 데이터 → CNN으로 결함 위치 패턴 분류

---

## 📊 데이터셋: WM-811K

| 항목 | 내용 |
|------|------|
| 출처 | Wu, M.J. et al., "Wafer Map Failure Pattern Recognition", IEEE TSM 2015 |
| 다운로드 | [Kaggle — WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) |
| 전체 샘플 | 811,457개 |
| 결함 레이블 보유 | 25,519개 (전체의 3.1%) |
| 결함 유형 | 8종 (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch) |

### 결함 패턴 분포 (클래스 불균형 심각)

| 패턴 | 샘플 수 | 비율 |
|------|---------|------|
| Edge-Ring | ~11,793 | 46.2% |
| Edge-Loc | ~5,189 | 20.3% |
| Center | ~4,294 | 16.8% |
| Loc | ~3,593 | 14.1% |
| Scratch | ~1,193 | 4.7% |
| Random | ~866 | 3.4% |
| Donut | ~555 | 2.2% |
| **Near-full** | **~149** | **0.6%** ← 가장 희귀 (Edge-Ring의 1/79) |

---

## 🧠 모델 구조 (SimpleCNN)

```
입력: (Batch, 1, 64, 64)  ← 그레이스케일 단일 채널, 64x64 균일 리사이징
  ↓
Conv Block 1: Conv2d(1→32) + BatchNorm + ReLU + MaxPool  →  (32, 32, 32)
Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool  →  (64, 16, 16)
Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool →  (128, 8, 8)
Conv Block 4: Conv2d(128→256) + BatchNorm + ReLU + MaxPool →  (256, 4, 4)
  ↓
Flatten → 4096
Dropout(0.5) → FC(4096→512) → ReLU → FC(512→8)
  ↓
출력: 8개 결함 클래스 확률
```

### 학습 전략
- **Weighted CrossEntropyLoss**: 희귀 클래스(Near-full)에 최대 79배 가중치 부여
- **Adam Optimizer**: lr=0.001
- **Early Stopping**: patience=7, 과적합 방지
- **Data Augmentation**: 좌우반전(50%), 상하반전(50%), 90° 무작위 회전

---

## 🏆 최종 성능 결과

**Balanced Accuracy: 92.02%**

| 결함 패턴 | F1-Score | 정밀도 | 재현율 |
|-----------|---------|--------|--------|
| **Edge-Ring** | **0.984** 🥇 | 0.980 | 0.988 |
| **Center** | **0.961** 🥈 | 0.961 | 0.961 |
| **Near-full** | **0.952** 🥉 | 1.000 | 0.909 |
| Random | 0.899 | 0.845 | 0.962 |
| Edge-Loc | 0.890 | 0.963 | 0.828 |
| Loc | 0.832 | 0.819 | 0.846 |
| Scratch | 0.828 | 0.756 | 0.916 |
| Donut | 0.827 | 0.731 | 0.952 |

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설치
```bash
pip install torch torchvision scipy scikit-learn matplotlib seaborn pandas numpy
```

### 2. 데이터 다운로드
[Kaggle — WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)에서 `LSWMD.pkl`을 다운로드하여 프로젝트 폴더에 저장합니다.

### 3. 실행 순서
```bash
# Step 1: 전처리 (최초 1회, 약 3~5분 소요)
python 01_dataset.py

# Step 2: CNN 모델 학습
python 03_train.py

# Step 3: 성능 평가 및 결과 그래프 생성
python 04_evaluate.py
```

---

## 📁 파일 구조

```
wm811k-cnn-project/
├── 01_dataset.py          # 데이터 전처리 및 PyTorch DataLoader
├── 02_cnn_model.py        # CNN 모델 아키텍처 정의
├── 03_train.py            # 모델 학습 (Early Stopping, Weighted Loss)
├── 04_evaluate.py         # 성능 평가 및 시각화
└── results/
    ├── fig1_label_distribution.png
    ├── fig2_wafer_samples.png
    ├── fig6_cnn_confusion_matrix.png  ⭐
    └── fig7_cnn_f1_scores.png         ⭐
```

---

## 🔬 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 딥러닝 프레임워크 | PyTorch 2.11.0 |
| 데이터 처리 | NumPy, Pandas, SciPy |
| 시각화 | Matplotlib, Seaborn |
| 평가 지표 | Scikit-learn |

---

## 🗺️ 프로젝트 로드맵

| Phase | 내용 | 상태 | 성능 |
|-------|------|------|------|
| Phase 1 | SECOM 센서 → XGBoost + AutoEncoder | ✅ 완료 | F1=0.7245, AUC=0.785 |
| **Phase 2** | **WM-811K → CNN 결함 분류** | ✅ **완료** | **Balanced Acc: 92.02%** |
| Phase 3 | NASA C-MAPSS → LSTM 예지보전 | ⬜ 예정 | — |

---

## 👥 팀: 리뉴얼 무적함대

관련 레포지토리: [secom-fdc-project](https://github.com/dongunny/secom-fdc-project)
