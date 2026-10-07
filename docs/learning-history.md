# 저장소별 학습 기록

[학습 기록 목차로 돌아가기](../README.md)

2023년 실습 파일과 Git 이력을 바탕으로 주제, 구현, 사용 도구와 기여 범위를 정리했습니다. 교육용 예제와 과제를 포함하며, 저장소 전체가 독자 설계한 코드라는 뜻은 아닙니다. 아래의 실행 결과는 노트북에 저장된 과거 출력입니다.

## 1. aiffel: 데이터 분석부터 생성형 모델까지

### 주요 구현

- **포켓몬 EDA:** Legendary 여부로 데이터를 나누고 타입·세대별 분포, 능력치 합계, 산점도와 피벗 테이블을 확인했습니다.
- **회귀:** 당뇨병 데이터에서 선형 모델, MSE, 기울기와 경사하강법을 함수로 구성했습니다. scikit-learn의 LinearRegression도 사용했고, 자전거 수요 데이터에서 날짜·시간 특성을 만들고 MSE·RMSE를 계산하는 코드를 작성했습니다.
- **분류:** 손글씨 숫자·와인·유방암 데이터에 Decision Tree, Random Forest, SVM, SGD Classifier, Logistic Regression을 적용하고 classification report와 confusion matrix로 비교했습니다.
- **카페 분석:** 판매·입퇴실 데이터에서 상품별 판매와 매출, 요금제별 금액, 고객 이용 시간과 소비의 관계를 탐색했습니다.
- **챗봇:** 한국어 질문·답변 전처리, subword 토큰화, 패딩, Transformer 인코더·디코더, 학습과 문장 생성 코드를 구성했습니다.
- **이미지 생성:** Stable Diffusion 1.5와 ControlNet을 이용해 Canny 윤곽선과 OpenPose 자세 조건을 다루는 실습을 진행했습니다.

### 도구와 자료

Python, Jupyter, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, tensorflow_datasets, PyTorch, Diffusers, OpenCV와 Pillow를 사용합니다. 실습마다 Pokemon.csv, scikit-learn 내장 데이터, Bike Sharing Demand, 카페 CSV, 질문·답변 CSV, 이미지와 사전 학습 모델을 사용했습니다.

### 확인 가능한 상태

분류 결과, 시각화, 동료 리뷰와 실험 과정이 남아 있습니다. 데이터 경로 중 13개 항목은 심볼릭 링크여서 데이터 원본을 별도로 준비해야 합니다.

챗봇에는 모델 요약, 10 epoch 학습 로그와 응답 예제가 저장되어 있습니다. 질문·답변 CSV는 저장소에 없고, 구성 요소의 정의 순서와 관련된 NameError도 함께 남아 있습니다.

ControlNet에는 이미지 출력이 저장된 셀이 있습니다. Canny 생성 셀의 중단 기록과 OpenPose 생성 부분에서 canny_pipe를 호출한 코드가 남아 있고, 두 조건을 결합하는 부분은 미완료입니다. 카페 분석에도 데이터 형식·실행 순서 관련 오류가 있어, 개별 탐색 결과와 전체 재현 여부를 구분해서 읽는 것이 좋습니다.

### 파일

- [포켓몬 EDA](../no0825_6_가랏몬스터볼전설의포켓몬찾아삼만리/포켓몬.ipynb)
- [당뇨병·자전거 회귀](../node0829/당뇨병,%20자전거갯수%20회귀분석.ipynb)
- [분류 모델 비교](../node0906/son_wine_cancer.ipynb) · [동료 리뷰](../node0906/README.md)
- [카페 분석](../node0915/cafe_analy.ipynb) · [동료 리뷰](../node0915/README.md)
- [Transformer 챗봇](../node1114/chatbot_practice.ipynb)
- [ControlNet](../node1115/proj1115.ipynb)

## 2. cdnn: PyTorch와 FashionMNIST

nn.Module을 상속한 신경망에 Flatten, Linear, ReLU를 쌓아 28×28 이미지를 10개 의류 범주로 분류했습니다. 모델 구조는 784 → 512 → 512 → 10인 완전연결 신경망입니다.

torchvision의 FashionMNIST를 DataLoader로 읽고 CrossEntropyLoss, SGD, 역전파, 파라미터 갱신과 평가 루프를 작성했습니다. 텐서 크기와 모델 파라미터를 출력하는 기초 실습도 포함되어 있습니다.

노트북에는 10 epoch 실행 로그와 마지막 테스트 정확도 70.4%, 평균 손실 0.791119가 저장되어 있습니다. 현재 환경에서 다시 측정한 수치와는 구분합니다.

- [학습·평가 노트북](https://github.com/prunusblssm/cdnn/blob/master/node1031/node1031.ipynb)

## 3. month9: 주택 가격 예측 베이스라인

가격, 방 수, 면적, 건축 연도와 위치 등의 변수를 가진 주택 데이터를 탐색했습니다. 결측값 확인, 날짜 가공, log1p 변환과 분포 시각화를 수행하는 코드가 있습니다.

GradientBoostingRegressor, XGBoost, LightGBM의 예측값을 산술평균하는 앙상블 함수와 교차검증 코드를 작성했습니다. pandas, NumPy, scikit-learn, missingno, Matplotlib, Seaborn을 함께 사용했습니다.

저장된 노트북에는 학습·테스트 파일을 반대로 읽는 코드와 price KeyError, KFold 설정 오류가 남아 있습니다. submission.csv는 있으나 현재 노트북과의 생성 경로를 검증하지 않았으므로 대회 성적이나 완료된 예측 파이프라인으로 소개하지 않습니다. 후속 node13.ipynb는 라이브러리 버전 확인과 데이터 읽기 단계까지 작성되어 있습니다.

- [주택 가격 베이스라인](https://github.com/prunusblssm/month9/blob/master/node0918/2019-ml-month-2nd-baseline.ipynb)
- [후속 실습](https://github.com/prunusblssm/month9/blob/master/node0919_questC/node13.ipynb)

## 4. Aiffel_repo: n면체 주사위

FunnyDice 클래스로 생성자, 인스턴스 변수와 메서드를 연습했습니다. throw, getval, setval로 던지기·조회·값 설정을 구현하고 콘솔 입력과 프로그램 시작점을 구성했습니다. Python 표준 라이브러리 random을 사용합니다.

기본 구현과 리뷰 체크리스트가 남아 있습니다. 입력값과 경계 조건을 모두 검증한 상태는 아니며, 예를 들어 setval에는 1보다 작은 값을 막는 처리가 없습니다.

- [main.py](https://github.com/prunusblssm/Aiffel_repo/blob/master/main.py)
- [당시 README](https://github.com/prunusblssm/Aiffel_repo/blob/master/README.md)

## 5. aiffel_project10-2-3-4: Python 기초 문제

한 파일에서 다음 세 문제를 연습했습니다.

1. 정규식으로 숫자 3자리-5자리-5자리 형식을 확인하고 마지막 다섯 자리를 #으로 가리기
2. 재귀 함수로 중첩 리스트 펼치기
3. 가변인수 중 10 이하인 숫자만 곱하기

문자열 슬라이싱, re 모듈, 반복문, 조건문, 재귀와 가변인수 사용 기록입니다. 전화번호 입력 부분은 종료 조건 없는 반복문으로 되어 있어 뒤의 예제는 따로 살펴보아야 합니다. 전화번호 형식은 과제에 맞춘 연습용 규칙입니다.

- [project10](https://github.com/prunusblssm/aiffel_project10-2-3-4/blob/main/project10)

## 6. aiffel-lsyo99: 동료의 주사위 코드 리뷰

원본은 [lsyo99/aiffel-quest](https://github.com/lsyo99/aiffel-quest)입니다. Python 주사위 구현은 원본 작성자의 작업이며, 제 계정에서 확인되는 기여는 python_dice/README.md의 리뷰입니다.

주사위의 동작과 주석을 검토하고, 1·2 이외의 모드 입력에 대한 보완 의견을 남겼습니다. 원본 구현 전체와 리뷰 기여를 구분해 보존합니다.

- [리뷰 문서](https://github.com/prunusblssm/aiffel-lsyo99/blob/master/python_dice/README.md)
- [리뷰 작성 커밋](https://github.com/prunusblssm/aiffel-lsyo99/commit/472d8cf9768a96514173da5c13851f14336de218)

## 7. code_Reiveiw_0906: 분류 실습 리뷰와 문서 협업

원본은 [Cellularhacker/aiffel](https://github.com/Cellularhacker/aiffel)이며 Python, 회귀와 분류 실습을 포함합니다. 제 계정의 커밋에서 확인되는 작업은 다음과 같습니다.

- 손글씨·와인·유방암 데이터의 다섯 분류 모델 비교 실습 리뷰
- 평가지표 비교 그래프, 주석, 회고와 가독성에 대한 의견 작성
- 공동 README에 AI 비서 관련 기사 요약과 당시 생각 추가
- 노트북 출력 줄바꿈 등 형식 수정

모델 구현은 원본 작성자의 작업으로 구분합니다. 아래 커밋에서 실제 문서·형식 변경 범위를 확인할 수 있습니다.

- [분류 실습 리뷰](https://github.com/prunusblssm/code_Reiveiw_0906/blob/main/NODE/1032/README.md)
- [주요 리뷰 커밋](https://github.com/prunusblssm/code_Reiveiw_0906/commit/69b2a9bdf6273bfcfb3dc0da59b881b77a37d1ab)
- [기사 요약 기여](https://github.com/prunusblssm/code_Reiveiw_0906/commit/153270b92e859ba0e6ff950bbdb09c53846e0af7)

## 8. aiffel_repo-1: Python·회귀 학습자료 포크

GitHub에서 [leejaesang6098/aiffel_repo](https://github.com/leejaesang6098/aiffel_repo)를 부모 저장소로 표시하는 포크입니다. 주사위, Python 기초 문제와 당뇨병·자전거 수요 회귀 실습이 있습니다. NumPy로 선형 모델·MSE·기울기를 구성하고 pandas·scikit-learn을 사용한 원본 자료를 살펴볼 수 있습니다.

확인한 기본 브랜치의 63개 커밋에서는 prunusblssm 계정의 별도 커밋이 확인되지 않았습니다. 따라서 원본 학습자료를 보존한 포크로 소개하며, 제 독자 구현이나 리뷰로 계산하지 않습니다.

- [회귀 실습](https://github.com/prunusblssm/aiffel_repo-1/blob/master/Quest0829/Quest_0829.ipynb)
- [주사위 구현](https://github.com/prunusblssm/aiffel_repo-1/blob/master/Quest1/Dice.py)
- [원본 작성자·리뷰어 기록](https://github.com/prunusblssm/aiffel_repo-1/blob/master/Quest1/README.md)

## 9. springkim623-aiffel: 첫 GitHub 협업 연습

[springkim623/aiffel](https://github.com/springkim623/aiffel)을 포크하고 README에 한 줄을 추가했습니다. 제 계정의 커밋에서 문서 수정이 확인됩니다. 기본 브랜치에는 README만 있으며, 포크와 문서 변경을 연습한 기록으로 남깁니다.

- [README](https://github.com/prunusblssm/springkim623-aiffel/blob/main/README.md)
- [본인 수정 커밋](https://github.com/prunusblssm/springkim623-aiffel/commit/22525bbb073e98cbc69ff2a9f1ee9e485cbab693)

---

이 문서는 기존 코드와 노트북을 수정하지 않고 학습 주제와 기여 범위를 안내하기 위해 추가했습니다.

