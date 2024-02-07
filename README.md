# 11가지 머신 러닝 모델 성능 테스트

## 공통 설정
1. Random Seed: 400
2. 데이터셋 분할 비율: Train 80%, Validation 20%
3. 수치형 결측값 처리: Mean 값 사용
4. 범주형 결측값 처리: 최빈값 사용

## 모델 리스트
1. DecisionTreeClassifier (기본 제공 코드)
2. RandomForestClassifier
3. Logistic Regression
4. Support Vector Machine
5. Gradient Boosting
6. Neural Networks (10 Epochs)
7. KNN
8. XGBoost
9. CatBoost
10. Voting Classifier (Hard) (DecisionTreeClassifier, XGBoost, Gradient Boosting)
11. Voting Classifier (Soft) (DecisionTreeClassifier, XGBoost, Gradient Boosting)

## 각 모델의 성능 결과
![Model Accuracies](https://github.com/justgotothedesk/LGAimers/assets/114928709/766cc02b-76df-47c0-83cc-a3df55c0c393)

