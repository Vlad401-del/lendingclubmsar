import pandas as pd
from sqlalchemy import create_engine

# Ganti dengan password database Supabase Anda
# Pastikan password yang mengandung karakter khusus di-URL-encode (misal @ menjadi %40)
DB_USER = "postgres.ddbepkvfyhgaikpzxelg"
DB_PASSWORD = "EA0supabase"
DB_HOST = "aws-1-ap-south-1.pooler.supabase.com" # Ganti dengan host Supabase Anda
DB_PORT = "6543"
DB_NAME = "postgres"

# Membuat connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    # Membuat engine koneksi
    engine = create_engine(DATABASE_URL)
    
    print("Berhasil tersambung ke Supabase!")
    
    # Contoh 1: Membaca tabel langsung menjadi Pandas DataFrame
    # Ganti 'nama_tabel_anda' dengan tabel yang ada di Supabase
    query = "SELECT * FROM dim_borrower LIMIT 10"
    df = pd.read_sql(query, engine)
    
    print("Preview Data:")
    print(df.head())

    # Contoh 2: Menyimpan DataFrame ke Supabase
    # df_hasil_ml.to_sql('nama_tabel_baru', engine, if_exists='replace', index=False)

except Exception as e:
    print("Gagal menyambung ke database:", e)