import pandas as pd

def calculate_missing_values(file_path):
    df = pd.read_csv(file_path)

    missing_data = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Missing Ratio': df.isnull().mean() * 100
    })

    print(missing_data)

if __name__ == "__main__":
    file_path = "/Users/shin/Downloads/submission.csv"

    calculate_missing_values(file_path)
