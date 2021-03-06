# 머신러닝을 이용한 빅데이터 분석 시스템 구현 


## 요  약
대규모 데이터를 분석하는 데이터로봇 시스템을 구현한다. 본 시스템은 Preview, Preprocessing, Evaluation으로 구성된다. 결측치 처리, 교차검증과 하이퍼파라미터 튜닝을 수행한다. 머신러닝에서는 여섯 가지의 분류와 회귀 알고리즘을 제공한다. 

각 알고리즘에 따라 최적의 하이퍼파라미터와 정확도, 정밀도, 재현율, RMSE, MSE를 결과로 제공한다.

## 1. 서론

 과거에 비해 현재의 데이터는 지속적으로 급격히 증가하는 대규모 데이터이다. 대규모로 저장된 데이터 안에서 상관관계를 계산하고 결과를 예측하거나 분류하는 데이터 분석이 주목받고 있다. 하지만 대규모 데이터를 사람이 계산하고 분석하기에는 한계가 있다. 
 
 그러므로 빅데이터 분석을 위해서는 데이터 분석 라이브러리 등을 활용한 프로그래밍을 이용하여 효율적으로 분석할 필요가 있다.
  우리는 데이터 분석 시스템인 일명 데이터로봇을 구현하기로 한다. 머신러닝의 지도학습 가운데 분류와 회귀모델을 이용하여 데이터 분석을 한다. 
  
  우리가 만들 데이터로봇에서는 머신러닝의 분류와 회귀 알고리즘들을 선택할 수 있고 선택한 알고리즘의 정확도 등을 보여준다. 데이터로봇은 파이썬 언어로 제작하며 사용자가 편리하게 사용할 수 있도록 PyQt5를 이용한 GUI환경을 구성한다.
  
  ## 2. 시스템 프로세스
<img width="50%" src="https://user-images.githubusercontent.com/68947314/170917153-9b3d517a-7275-4946-aa44-93e0cea783ae.png" />
  
  
  본 데이터로봇의 시스템 프로세스는 그림1과 같다. 
   먼저 데이터들을 살펴보는 Preview 단계, 데이터의 전처리를 할 수 있는 PreProcessing 단계 그리고 알고리즘을 선택해서 학습하고 각 알고리즘의 정확도 등을 볼 수 있는 Evaluation 단계로 나뉜다.
  
  ## 3. 데이터로봇 시스템
  본 데이터로봇 시스템은 Preview 단계, Preprocessing 단계, 그리고 Evaluation 단계로 구성된다.
  
  
  ### Preview 단계
  
  <img width="50%" src="https://user-images.githubusercontent.com/68947314/170917676-200fb1c6-8322-4918-ae43-dcfb120be2f7.png" />
  
  그림 2의 Preview단계에서 사용한 예제 데이터는 Kaggle의 심장병 데이터셋이다. QFileDialog의 getOpenFileName()을 이용한 Open File 버튼을 누르면 csv파일을 열 수 있다.      Current relation을 살펴보면 파일명, 데이터의 행과 열의 개수를 알 수 있다. Attributes 그룹에는 데이터의 속성들을  볼 수 있다. 
  
  
  이 속성을 선택하면 그에 따른 히스토그램을 차트로 볼 수 있다. 히스토그램은 데이터의 분포를 한눈에 알아보기 쉽게 한다. dataset에는 데이터의 초기 10개 값을 확인 할 수 있고 기초통계(개수, 평균, 표준편차, 사분위수 등)를 보여준다. 데이터로봇의 첫 화면인 Preview 단계는 데이터를 전반적으로 살펴보는 단계이다. 
  
 
 ### PreProcess 단계
 
 
 <img width="50%" src="https://user-images.githubusercontent.com/68947314/170917757-03e3a28a-1be0-4f12-bf51-d06a300702de.png" />
 
 
 다음 단계인 데이터 전처리를 위해 Preprocessing탭을 누르면 그림3과 같이 나타난다. 먼저 속성에 결측치가 있으면 결측치의 개수를 보여준다. 결측치가 있으면 제대로 된 학습을 할 수 없기에 이를 처리할 필요가 있다. 본 데이터로봇에서는 delete, mean, median, most_frequent 등의 결측치 처리 방법을 사용한다.
 
 
  delete는 결측치가 있는 행을 삭제하는 방법이다. 결측치의 특성이 무작위로 손실되지 않았다면, 대부분의 경우 가장 좋은 방법은 제거하는 것이다. 그러나 데이터셋의 크기가 작고 결측치가 많으면 결측치 대체방법이 필요하다. mean은 평균값으로 대체하는 것이다. 
  
  해당 값으로 대치 시 변수의 평균값이 변하지 않는다는 장점이 있지만, 많은 단점이 존재한다. median은 숫자형에서 결측 값을 제외한 중앙값으로 대치하는 방법이다. 중간 값은 모든 관측 값을 이용하지 않으므로 평균값보단 이상치의 영향을 덜 받는다. most_frequent는 가장 빈번히 나온 값으로 대체한다. 하지만 데이터셋의 통계 분석에 영향을 줄 수 있는 단점이 존재한다. 
  
  
  그림3 안의 아래에는 박스 플롯을 그려서 데이터의 분포를 시각화한다. 박스 플롯은 직관적이며 이상치를 살펴보는데 유용하다. 만약 결측치가 있는 열이 있으면 박스플롯에 표시되지 않는다. 결측치를 처리하면 Missing Value에 있는 숫자와 박스 플롯이 그에 맞게 바뀐다.
  
  
  또한 교차검증을 사용한다. 데이터 개수가 적은 데이터 셋에 대한 정확도를 향상하고 과소적합 등의 문제를 해결한다. 데이터로봇에서는 기본적으로 선택이 되어 있지 않으며 사용자가 원하면 5-fold, 10-fold를 선택할 수 있다(그림3의 k-fold).
  마지막으로 변수들 사이의 값의 크기 차이로 인한 왜곡이 생길 수 있기 때문에 StandardScaler를 이용해 데이터 표준화를 하도록 하였다.
  
  ### Evaluation 단계
  
  <img width="50%" src="https://user-images.githubusercontent.com/68947314/170917834-153b1f35-0db0-4e0c-9e01-f9974458a624.png" />
  
  
  그림4의 Evaluation탭에서는 분류나 회귀 알고리즘을 선택할 수 있도록 하였다. 먼저 데이터의 종속변수를 선택할 수 있다. 기본적으로 마지막 열이 종속변수로 많이 쓰이기 때문에 마지막 열을 default로 설정하였다. 사용자가 원하면 종속변수를 바꿀 수 있다.
  
  
  본 데이터로봇에 넣은 알고리즘에는 K-Nearest Neighbor, Linear Regression, Ridge Regression, Lasso Regression, Logistic Regression, Decision Tree가 있다. 알고리즘 선택을 간편히 하도록 All, Classification, Regression 버튼으로 나눠 그에 맞는 알고리즘이 선택되도록 하였다.
  
 <img width="50%" src="https://user-images.githubusercontent.com/68947314/170917914-90b257c9-6b64-4478-981d-81e30d631ea8.png" />
 
 
 그림 5는 Decision Tree를 선택했을 때 나타나는 화면이다. Decision Tree의 속성들을 사용자가 직접 입력할 수 있게 만들었다. splitter는 기본적으로 Best가 선택되게 했다. 만약 숫자들을 고치지 않고 버튼을 누르면 속성들에 맞는 최솟값으로 학습하게 하였다.
 
 <img width="50%" src="https://user-images.githubusercontent.com/68947314/170917984-e6dc380e-0d22-47b5-a854-f7fff5679d91.png" />
 
 
그림 6은 선택한 머신러닝 알고리즘으로 학습한 결과화면이다. 만약 교차검증을 선택하면 하이퍼파라미터 튜닝을 통해 최적의 파라미터를 선택해서 결과를 화면에 보여준다. 분류 알고리즘을 선택하면 정확도, 정밀도, 재현율을 보여주고 회귀 알고리즘을 선택하면 정확도, RMSE, MSE를 보여준다. 

Decision Tree는 분류와 회귀 모두 가능하여 여기서는 모든 결과를 보이도록 해 보았다.
  알고리즘들을 여러 개 선택해서 정확도를 서로 비교하면 데이터셋에 알맞은 알고리즘들을 판단할 수 있다. 현재 우리가 예로 사용한 심장병 데이터셋의 타깃 데이터는 이진형이므로 분류 알고리즘만을 선택하면 된다. 본 논문의 지면에서는 편의상 모두 선택한 결과를 보인다.
  
  분석하고자 하는 데이터셋의 반응변수의 값이 연속형이면 회귀 알고리즘을 선택하면 된다. 
  
  
 ## 4. 결론
 
 
 본 연구에서는 대량의 데이터를 분석하기 위해서 데이터로봇 시스템을 개발하였다. 본 시스템은  Preview 단계, Preprocesssing 단계, 그리고 Evaluation 단계로 처리된다. Preview 단계에서는 사용자가 선택한 데이터셋에 대한 전반적인 정보를 제공한다. 
 
 PreProcessing 단계에서는 지정한 데이터셋 내의 결측치를 확인할 수 있다. 또한 결측치를 처리하기 위해 delete, mean, median, most_frequent와 같은 방법을 제공한다. 이 외에도 서로 다른 변수의 값 범위를 일정한 범위로 조정하는 표준화 기능을 제공한다. 마지막 단계인 Evaluation에서는 분류와 회귀 알고리즘을 선택할 수 있다.
 
 이 단계에서는  사용자가 원하는 종속변수와 알고리즘을 이용하여 데이터를 분석하고 결과를 제공한다. 데이터 분석 알고리즘으로는 K-Nearest Neighbor, Linear Regression, Ridge Regression, Lasso Regression, Logistic Regression, Decision Tree를 제공한다. 최적의 모델을 생성하기 위해서 교차검증과 하이퍼파라미터 튜닝을 지원한다. 
 
 학습을 위하여 분류 알고리즘을 선택하면 정확도, 정밀도, 재현율을 보여주고 회귀 알고리즘을 선택하면 정확도, RMSE, MSE를 보여준다.
 
 
  
## 참고문헌
[1] Weka Knowledge Explorer,
https://www.cs.waikato.ac.nz/~ml/weka/gui_explorer.html


[2] 클린턴 브라운리, 파이썬 데이터 분석 입문, 한빛미디어, 2017


[3] 권철민, 파이썬 머신러닝 완벽 가이드, 위키북스, 2020


[4] DATADOCTOR,
https://datadoctorblog.com/2020/12/31/Br-ML-decision-tree/
