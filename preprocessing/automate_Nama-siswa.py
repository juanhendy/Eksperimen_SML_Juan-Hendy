import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    # 1. Path Configuration
    input_path = 'namadataset_raw/default of credit card clients.xls'
    output_dir = 'preprocessing/namadataset_preprocessing'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Memulai proses preprocessing...")

    # 2. Load Data
    # Menggunakan xlrd karena format .xls
    df = pd.read_excel(input_path, header=1)

    # 3. Cleaning & Transformation
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    
    df.rename(columns={'default payment next month': 'Y'}, inplace=True)

    # Sinkronisasi Kategori (Education & Marriage)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    df['MARRIAGE'] = df['MARRIAGE'].replace([0], 3)

    # 4. Handling Outliers (Clipping)
    cols_to_clip = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    for col in cols_to_clip:
        upper_limit = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    # 5. Scaling
    X = df.drop(['Y'], axis=1)
    y = df['Y']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Gabungkan kembali menjadi satu dataframe untuk disimpan
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['target'] = y.values

    # 6. Save Output
    output_file = os.path.join(output_dir, 'cleaned_data.csv')
    df_final.to_csv(output_file, index=False)
    
    print(f"Berhasil! Data tersimpan di: {output_file}")

if __name__ == "__main__":
    run_preprocessing()