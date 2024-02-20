import pandas as pd
from sklearn.impute import SimpleImputer

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
    df_with_missing = df[df[numerical_column].isnull()]
    X_train, X_test, y_train, y_test = train_test_split(
        df_non_missing[numerical_columns + categorical_columns],
        df_non_missing[numerical_column],
        test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train[numerical_columns + categorical_columns], y_train)
    predicted_values = model.predict(df_with_missing[numerical_columns + categorical_columns])
    df.loc[df[numerical_column].isnull(), numerical_column] = predicted_values

# 범주형 변수 결측치 채우기 (최빈값 사용)
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

# 결과 출력
print(df)
