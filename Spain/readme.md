# 🔥 스페인 산불 연료 분류 MLP 분류기

이 프로젝트는 스페인 산불 지역의 연료 유형을 분류하기 위해 MLP(Multi-Layer Perceptron) 기반 분류 모델을 구현한 것입니다.  
기초 데이터 전처리부터 클러스터링, 모델 학습, 새로운 데이터 예측까지의 전체 파이프라인을 다룹니다.

> 참고 논문: **"Fuel type mapping in Spain using Sentinel-2 and machine learning techniques"**  
> (해당 논문에서 사용된 feature 구성과 MLP 모델 구조를 참고하였습니다)

---

## 📁 프로젝트 구조

📦 project-root
├── resource/
│ └── example.csv # 예측할 입력 데이터
├── result/
│ └── example_clustered.csv # 예측된 클러스터 결과 저장
├── Spain_Fule_MLP/
│ ├── mlp_fuel_model.pth # 학습된 모델 파일
│ └── scaler.pkl # 학습 시 사용된 스케일러
├── cluster_kmeans.py # KMeans 클러스터링 (클래스 레이블 생성)
├── train_mlp.py # MLP 모델 학습 및 저장
├── predict_mlp.py # 새 데이터 예측 스크립트
└── README.md



---

## 🚀 실행 순서

1. ### 1️⃣ 클러스터 라벨 생성 (KMeans)

- `./resource/example.csv` 파일을 읽어들여 KMeans로 클러스터링 수행
- `./result/example_clustered.csv`에 클러스터 라벨 저장 (`cluster` 컬럼)

2. ### 2️⃣ MLP 모델 학습

- 클러스터된 데이터를 기반으로 MLP 분류기 학습
- 학습된 모델(`mlp_fuel_model.pth`)과 스케일러(`scaler.pkl`) 저장

3. ### 3️⃣ 새로운 데이터 예측

- `./resource/example.csv`에 있는 새 데이터를 기반으로 연료 클러스터 예측
- 예측 결과는 `./result/example_clustered.csv`에 저장

---

## 📌 주요 기능 설명

| 파일명 | 설명 |
|--------|------|
| `cluster_kmeans.py` | 클러스터링 수행 (KMeans, 라벨 0~9 부여) |
| `train_mlp.py`      | MLP 모델 정의 및 학습, 스케일러 저장 |
| `predict_mlp.py`    | 학습된 모델을 불러와 새 데이터 예측 수행 |

---

## 📊 사용된 Feature 목록

- Sentinel-2 밴드: `b01` ~ `b12`, `b8A`
- Vegetation index: `NDVI`, `EVI`, `GNDVI`, `BSI`, `SAVI`, `NDWI`, `NDMI`, `NBR`, `GCI`
- 기타: `AVI`, `VARI`

---

## ⚠️ 주의 사항

- `example.csv` 파일에는 위에서 언급한 모든 feature가 포함되어 있어야 합니다.
- 모델 학습 시 사용한 스케일러(`scaler.pkl`)는 반드시 예측 단계에서도 동일하게 사용되어야 합니다.
- 데이터 파일은 GitHub에 업로드하지 않았으며, 사용자 본인의 데이터를 활용해 실행해야 합니다.

---

## ✅ 예시 결과

| x | y | ndvi | evi | gci | ... | predicted_cluster |
|---|---|------|-----|-----|-----|-------------------|
| ... | ... | ... | ... | ... | ... | 3 |
| ... | ... | ... | ... | ... | ... | 7 |

---

## 📚 참고

- 논문: Fuel type mapping in Spain using Sentinel-2 and machine learning techniques
- PyTorch, Pandas, Scikit-learn 등 주요 라이브러리 사용

---

## 🧑‍💻 개발자

- 성결대학교 컴퓨터공학과 유진영 (20220852)