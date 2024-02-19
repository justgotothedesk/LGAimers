import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')

# 결측치가 없는 열들 선택
non_missing_columns = ['bant_submit', 'business_unit', 'customer_idx', 'enterprise', 'lead_desc_length',
                        'customer_position', 'response_corporate', 'ver_cus', 'ver_pro']

# 결측치가 없는 열들로만 이루어진 데이터프레임 생성
df_non_missing = df[non_missing_columns].dropna()

# 결측치를 채울 열들 선택
columns_to_impute = ['customer_type', 'historical_existing_cnt', 'id_strategic_ver', 'it_strategic_ver',
                      'idit_strategic_ver', 'customer_job', 'inquiry_type', 'product_category',
                      'product_subcategory', 'product_modelname', 'expected_timeline', 'ver_win_rate_x',
                      'ver_win_ratio_per_bu', 'business_area', 'business_subarea']

for column_with_missing_values in columns_to_impute:
    # 결측치가 있는 열을 제외한 열들을 특성으로 선택
    features = non_missing_columns

    # 결측치를 채울 열과 그 외 열로 데이터 분할
    df_missing = df[df[column_with_missing_values].isnull()]

    # 특성과 타겟 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        df_non_missing[features],
        df_non_missing[column_with_missing_values],
        test_size=0.2, random_state=42
    )

    # 선형 회귀 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 학습된 모델을 사용하여 결측치 예측
    predicted_values = model.predict(df_missing[features])

    # 예측값을 데이터프레임에 적용
    df.loc[df[column_with_missing_values].isnull(), column_with_missing_values] = predicted_values

# 결과 출력
print(df)
