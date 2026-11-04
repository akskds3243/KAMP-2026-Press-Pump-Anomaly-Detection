# Press Pump Anomaly Detection

**진동·전류 시계열 기반 프레스 유압펌프 이상탐지 및 현장 의사결정 지원 시스템**

제6회 K-인공지능 제조데이터 분석 경진대회 출품 프로젝트입니다. 프레스 유압펌프의 진동·전류 센서 데이터를 활용하여 설비 이상 상태를 탐지하고, 오경보(FP) 및 미탐지(FN) 발생 조건을 분석하여 현장 진단과 경보 의사결정을 지원하는 AI 시스템을 구현했습니다.

**핵심 목표:** 단순한 모델 성능 향상을 넘어, 실제 제조데이터의 시간적 특성을 반영하고 모델의 오류분석 결과를 현장 경보 안정화 및 점검 의사결정에 연결하는 것입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | Press Pump Anomaly Detection |
| 분야 | Manufacturing AI / Anomaly Detection / Predictive Maintenance |
| 데이터 | 프레스 유압펌프 진동 2채널 및 모터 전류 1채널 |
| 분석 모델 | Isolation Forest, Autoencoder, XGBoost |
| 보조 지표 | Hybrid Anomaly Score |
| 경보 후처리 | Temporal Persistence |
| 현장 UI | Gradio |
| 개발환경 | Python / Google Colab / Jupyter Notebook |

### 해결하고자 한 문제

- 실제 이상 상태임에도 센서 변화가 작아 모델이 정상으로 판단하는 미탐지(FN)
- 정상 운전 중 일시적인 진동·전류 변동으로 인해 발생하는 오경보(FP)
- 모델의 단순 정상·이상 판정만으로는 점검 우선순위를 판단하기 어려운 문제

이를 해결하기 위해 데이터 전처리부터 모델 비교, 오류분석, 경보 안정화 및 현장 UI 구현까지 하나의 분석 파이프라인으로 구성했습니다.

## 2. 데이터 이해 및 전처리

### 데이터 구성

| 구분 | 원본 데이터 | 정제 후 |
|---|---:|---:|
| 정상 데이터 | 20,000 | 19,999 |
| 이상 데이터 | 600 | 600 |
| 합계 | 20,600 | 20,599 |

정상 데이터와 이상 데이터는 서로 다른 날짜에 수집된 독립적인 측정 세션입니다. 따라서 두 데이터를 하나의 정상 상태에서 고장으로 진행되는 연속 시계열로 연결하지 않고, 각 세션 내부의 시간적 순서를 유지했습니다.

### 주요 전처리 과정

1. 저장용 인덱스 컬럼 및 중복 측정 1건 제거
2. TimeStamp 기준 데이터 정렬
3. 0.5초 초과 측정 공백에서 Segment 분리
4. Segment 내부에서만 Rolling Feature 생성
5. 5개의 연속 관측값이 확보된 이후부터 특징 계산
6. Equipment_state를 정답 라벨로만 사용하여 Target Leakage 방지
7. 시간순 Train 60% / Validation 20% / Test 20% 분할
8. 정상 Train 데이터 기준 StandardScaler 표준화

### 시계열 특징 생성

사용한 센서는 다음과 같습니다.

- AI0_Vibration: 진동 센서
- AI1_Vibration: 진동 센서
- AI2_Current: 모터 전류 센서

각 센서에서 다음 5개 특징을 구성했습니다.

- Raw: 원본 센서값
- Rolling Mean: 최근 관측값의 평균
- Rolling STD: 최근 관측값의 변동 정도
- RMS: 신호 크기
- PTP: 최근 관측값의 최대값과 최소값 차이

총 **3개 센서 × 5개 특징 = 15개 입력 Feature**를 생성했습니다.

측정 공백을 임의로 보간하지 않고 새로운 Segment로 구분하여 실제 관측되지 않은 데이터가 모델 입력에 포함되지 않도록 구성했습니다.

## 3. Window Size 선정

시계열 분석에 사용할 Window Size를 결정하기 위해 3-Step, 5-Step, 10-Step을 Validation 데이터에서 비교했습니다.

| Window Size | XGBoost F1 | Isolation Forest F1 |
|---|---:|---:|
| 3-Step | 0.7006 | 0.2818 |
| 5-Step | 0.8023 | 0.3155 |
| 10-Step | 0.7887 | 0.4138 |

Isolation Forest는 10-Step에서 가장 높은 Validation F1-score를 기록했습니다.

그러나 10-Step은 진단을 위해 더 많은 연속 관측값이 필요합니다. 반면 5-Step은 XGBoost에서 가장 높은 성능을 보였으며 Isolation Forest 역시 3-Step 대비 개선됐습니다.

따라서 모델 성능과 현장 반응성을 함께 고려하여 **5-Step을 최종 선택**했습니다.

## 4. 다중 AI 모델 구성

### Isolation Forest

- 정상 Train 데이터만 학습
- 정상 데이터 분포에서 벗어난 패턴을 탐지
- 최종 Test에서 가장 높은 Recall과 F1-score 기록
- 최종 주 이상탐지 모델로 선정

### Autoencoder

- 정상 Train 데이터만 학습
- 정상 패턴을 압축하고 복원하는 신경망
- 원본 입력과 복원 결과의 차이인 MSE를 계산
- 정상 패턴으로부터의 이탈 정도를 보조 지표로 활용

### XGBoost

- 정상·이상 Train 데이터를 함께 학습
- 정상과 이상을 구분하는 지도학습 모델
- 이상 클래스의 예측확률 산출
- 보조 판별모델로 활용

### Hybrid Anomaly Score

XGBoost의 이상 클래스 예측확률과 정규화된 Autoencoder 이상점수를 동일한 가중치로 결합했습니다.

**Hybrid Anomaly Score = 0.5 × XGBoost 이상확률 + 0.5 × 정규화 AE Score**

이 방식은 본 프로젝트에서 선택한 가중합 구조이며, Hybrid Anomaly Score 자체가 통계적으로 보정된 이상확률을 의미하지는 않습니다.

Validation 데이터에서 F1-score가 최대가 되는 지점을 탐색하여 **Hybrid 판단 임계값을 0.3278**로 설정했습니다.

## 5. 모델 성능 비교

동일한 Test 데이터에서 네 모델을 비교했습니다.

| Model | Precision | Recall | F1-score | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.8228 | 0.6250 | 0.7104 | 0.9898 | 14 | 39 |
| Autoencoder | 0.9362 | 0.4231 | 0.5828 | 0.9942 | 3 | 60 |
| XGBoost | 1.0000 | 0.3654 | 0.5352 | 0.9524 | 0 | 66 |
| Hybrid Anomaly Score | 0.9348 | 0.4135 | 0.5733 | 0.9943 | 3 | 61 |

Isolation Forest는 Recall 0.6250, F1-score 0.7104를 기록하여 비교 모델 중 가장 높은 이상탐지 성능을 나타냈습니다.

특히 실제 이상 상태를 놓친 FN이 39건으로 가장 적었습니다.

따라서 Isolation Forest를 주 이상탐지 모델로 선정하고, Autoencoder와 XGBoost 및 Hybrid Anomaly Score를 보조 진단정보로 활용했습니다.

## 6. FP/FN 오류분석

모델의 평균 성능뿐 아니라 어떤 조건에서 잘못 판단하는지를 확인하기 위해 TP/FN 및 TN/FP의 주요 센서 특징을 비교했습니다.

### 미탐지(FN) 분석

실제 이상 상태 중 탐지에 성공한 TP와 놓친 FN을 비교했습니다.

| Feature | TP 평균 | FN 평균 |
|---|---:|---:|
| AI0_Vibration_rms | 0.3442 | 0.0527 |
| AI1_Vibration_rms | 0.1947 | 0.0719 |
| AI2_Current_rms | 275.0539 | 183.3187 |

FN에서는 TP보다 진동 크기와 일부 전류 관련 특징이 상대적으로 작게 나타났습니다.

이는 실제 이상 상태라도 센서 특징이 정상 패턴과 유사한 경우 모델이 이상을 놓칠 가능성이 있음을 시사합니다.

### 오경보(FP) 분석

정상 상태 중 정상으로 판정한 TN과 이상으로 잘못 판정한 FP를 비교했습니다.

| Feature | TN 평균 | FP 평균 |
|---|---:|---:|
| AI0_Vibration_rms | 0.0479 | 0.1144 |
| AI1_Vibration_ptp | 0.1587 | 0.5225 |
| AI2_Current_rms | 94.5812 | 182.9479 |

FP에서는 진동과 전류 관련 특징이 TN보다 크게 나타났습니다.

이는 정상 운전 중 일시적인 센서 변동이 이상 패턴과 유사하게 나타날 때 오경보가 발생할 수 있음을 시사합니다.

이 분석은 모델의 오류 발생 특성을 파악한 것이며, 특정 기계적 고장 원인을 확정한 것은 아닙니다.

## 7. Temporal Persistence 기반 경보 안정화

단일 시점의 이상 신호가 곧바로 최종 경보로 연결되는 것을 방지하기 위해 Temporal Persistence Rule을 적용했습니다.

**최근 5개 Hybrid 판단 중 3개 이상이 이상이면 최종 경보 발생**

측정 공백으로 Segment가 변경되면 이전 경보 판단 기록은 초기화됩니다.

### 적용 전후 성능

| 지표 | 적용 전 | 적용 후 |
|---|---:|---:|
| Precision | 0.9348 | 1.0000 |
| Recall | 0.4135 | 0.3558 |
| F1-score | 0.5733 | 0.5248 |
| FP | 3 | 0 |
| FN | 61 | 67 |

Persistence 적용으로 FP는 3건에서 0건으로 감소했습니다.

그러나 FN은 61건에서 67건으로 증가했고 Recall과 F1-score도 감소했습니다.

따라서 Persistence를 탐지 성능 자체를 높이는 기술이 아니라, **오경보 감소와 탐지 민감도 간 Trade-off를 조절하는 경보 안정화용 후처리**로 활용했습니다.

## 8. Gradio 기반 현장 진단 UI

모델 예측 결과를 실제 작업자의 점검 판단에 연결하기 위해 Gradio 기반 대시보드를 구현했습니다.

### Mode 1 — 최근 5개 관측값 진단

- Isolation Forest 정상/이상 판정
- Autoencoder 복원오차(MSE)
- Hybrid Anomaly Score
- 센서별 추세 그래프
- AE 재구성오차 기반 센서별 상대적 이상 기여도
- 현장 점검 가이드

### Mode 2 — 정상·이상 세션 연속 모니터링

- 정상 및 이상 세션별 모델 추론
- 시간순 이상 위험도 추이 확인
- Persistence 최종 경보 분석
- 오경보 억제 및 지속 이상상태 감시 검증

### Mode 3 — 현장 모니터링 시뮬레이션

- 센서 스트림이 유입되는 상황을 가정한 UI
- 시점별 이상탐지 결과 표시
- Hybrid 위험도 및 Persistence 상태 확인
- 최종 경보와 현장 점검 안내 표시

센서별 이상 기여도는 점검 우선순위를 지원하는 보조정보이며, 실제 고장 원인을 확정하는 지표는 아닙니다.

## 9. 실행 방법

### 사용 기술

- Python
- Pandas / NumPy
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib
- Gradio

### 실행 순서

1. Repository에서 `press_pump_anomaly_detection.ipynb` 파일을 다운로드하거나 Google Colab에서 엽니다.
2. `requirements.txt`에 명시된 라이브러리를 설치합니다.
3. 데이터 이용 조건에 따라 준비한 `press_data_normal.csv`, `outlier_data.csv`를 Notebook 실행 경로에 배치합니다.
4. Notebook의 Cell을 위에서부터 순서대로 실행합니다.
5. 데이터 전처리, 모델 학습, Test 성능평가, FP/FN 분석, Persistence 후처리, Gradio UI 순서로 결과를 확인합니다.

최종 예측결과는 `prediction_result.csv`로 생성됩니다.

실험은 Google Colab Python 3.13.15 환경을 기준으로 진행했습니다. 라이브러리 버전과 실행환경에 따라 세부 결과가 달라질 수 있습니다.

## 10. 프로젝트 차별성

본 프로젝트의 주요 차별점은 단순히 여러 AI 모델을 사용한 것이 아니라, 실제 제조데이터의 구조와 오류 특성을 분석하고 이를 현장 진단 시스템 설계에 반영한 점입니다.

- Target Leakage 방지를 위한 정답 라벨 제외
- 측정 공백을 고려한 Segment 기반 시계열 처리
- Validation 비교를 통한 5-Step 선정
- IF, AE, XGBoost의 역할 분담
- TP/FN 및 TN/FP 특징량 기반 오류분석
- Temporal Persistence를 통한 최종 경보 안정화
- Gradio 기반 현장 의사결정 지원 UI 구현

데이터 진단 → 시계열 특징 생성 → 다중 모델 학습 → 오류분석 → 경보 안정화 → 현장 UI로 이어지는 분석 흐름을 구현했습니다.

## 11. 프로젝트 한계 및 개선 방향

본 프로젝트는 제공된 제조데이터를 기반으로 수행되었으므로 다음과 같은 한계가 있습니다.

- 정상 및 이상 데이터가 독립적인 세션으로 구성되어 있어 실제 고장 발생 몇 초 전 탐지 성능을 검증하지 못했습니다.
- 제공된 데이터와 분할 조건에서 얻은 결과이므로 다른 설비 및 운전 조건에서의 일반화 성능은 추가 검증이 필요합니다.
- Hybrid Anomaly Score는 보정된 이상확률이 아닌 종합 위험도 점수입니다.
- Persistence 적용 시 오경보는 감소하지만 미탐지가 증가할 수 있습니다.
- Gradio UI는 현장 적용을 가정한 프로토타입이며 실제 설비에 직접 연결해 운용한 결과는 아닙니다.

향후에는 다양한 설비 및 운전 조건의 데이터를 추가 확보하고, 이상 위험도 임계값과 Persistence 조건을 개선하여 현장 적용성을 높이는 방향으로 확장할 수 있습니다.


---

**Project Type:** Manufacturing AI Competition Project

**Keywords:** Manufacturing AI, Time Series, Anomaly Detection, Isolation Forest, Autoencoder, XGBoost, Temporal Persistence, Gradio
