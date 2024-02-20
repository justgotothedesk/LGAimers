import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# 데이터 로드
df = pd.read_csv('train.csv')

# 수치형 변수 선택
numerical_columns = ['bant_submit', 'com_reg_ver_win_rate', 'historical_existing_cnt',
                     'lead_desc_length', 'ver_cus', 'ver_pro', 'ver_win_rate_x', 'ver_win_ratio_per_bu']

# 범주형 변수 선택
categorical_columns = ['business_unit', 'customer_type', 'enterprise', 'customer_job', 'inquiry_type',
                       'product_category', 'product_subcategory', 'product_modelname', 'expected_timeline',
                       'business_area', 'business_subarea', 'response_corporate']

# 결측치가 없는 열들로만 이루어진 데이터프레임 생성
df_non_missing = df[numerical_columns + categorical_columns].dropna()

# 수치형 변수 결측치 채우기
for numerical_column in numerical_columns:
    df_with_missing_numerical = df[df[numerical_column].isnull()]
    X_train_numerical, X_test_numerical, y_train_numerical, y_test_numerical = train_test_split(
        df_non_missing[numerical_columns + categorical_columns],
        df_non_missing[numerical_column],
        test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train_numerical[numerical_columns + categorical_columns], y_train_numerical)
    predicted_values_numerical = model.predict(df_with_missing_numerical[numerical_columns + categorical_columns])
    df.loc[df[numerical_column].isnull(), numerical_column] = predicted_values_numerical

# 범주형 변수 결측치 채우기 (CatBoost 사용)
for categorical_column in categorical_columns:
    df_with_missing_categorical = df[df[categorical_column].isnull()]
    X_train_categorical, X_test_categorical, y_train_categorical, y_test_categorical = train_test_split(
        df_non_missing[numerical_columns + categorical_columns],
        df_non_missing[categorical_column],
        test_size=0.2, random_state=42
    )
    catboost_model = CatBoostClassifier(
        verbose=0,
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        one_hot_max_size=5
    )
    catboost_model.fit(X_train_categorical[numerical_columns + categorical_columns], y_train_categorical, cat_features=categorical_columns)
    predicted_values_categorical = catboost_model.predict(df_with_missing_categorical[numerical_columns + categorical_columns])
    df.loc[df[categorical_column].isnull(), categorical_column] = predicted_values_categorical

# 결과 출력
print(df)
