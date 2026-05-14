import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    # 1. Konfigurasi Path
    # Pastikan folder ini sesuai dengan struktur repositori Kriteria 1 Anda
    input_path = 'namadataset_raw/default of credit card clients.xls'
    output_dir = 'preprocessing/namadataset_preprocessing'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Memulai Automasi Preprocessing ---")

    # 2. Load Data
    # Dataset ini menggunakan baris ke-2 sebagai header
    df = pd.read_excel(input_path, header=1)

    # 3. Data Cleaning (Konsisten dengan Eksperimen)
    # Hapus kolom ID
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    
    # Rename Target
    df.rename(columns={'default payment next month': 'target'}, inplace=True)

    # Sinkronisasi Kategori (Mengganti nilai pada kolom asli)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    df['MARRIAGE'] = df['MARRIAGE'].replace([0], 3)

    # 4. Penanganan Outlier (Clipping)
    cols_to_clip = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    for col in cols_to_clip:
        upper_limit = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    # 5. Penskalaan Fitur (Scaling)
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    
    # Menggabungkan kembali menjadi satu DataFrame agar mudah disimpan ke CSV
    df_final = pd.DataFrame(X_scaled_array, columns=X.columns)
    df_final['target'] = y.values

    # 6. Simpan Output
    output_file = os.path.join(output_dir, 'cleaned_data.csv')
    df_final.to_csv(output_file, index=False)
    
    print(f"Berhasil! File tersimpan di: {output_file}")
    print(f"Jumlah baris: {df_final.shape[0]}, Jumlah fitur: {df_final.shape[1]-1}")

if __name__ == "__main__":
    run_preprocessing()