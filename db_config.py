# ==========================================================
# db_config.py
# Helper Koneksi Database Supabase (Shared Module)
# Digunakan oleh seluruh skrip ML (01-04)
# ==========================================================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def get_engine():
    """
    Membuat SQLAlchemy engine ke Supabase PostgreSQL.
    Mengambil DATABASE_URL dari file .env
    """
    DATABASE_URL = os.environ.get("DATABASE_URL")

    if not DATABASE_URL:
        raise ValueError(
            "[ERROR] DATABASE_URL tidak ditemukan di file .env!\n"
            "Pastikan file .env berisi:\n"
            'DATABASE_URL="postgresql://postgres.xxxx:password@host:6543/postgres"'
        )

    # Tambahkan SSL requirement untuk Supabase
    if "?" not in DATABASE_URL:
        DATABASE_URL += "?sslmode=require"
    elif "sslmode=" not in DATABASE_URL:
        DATABASE_URL += "&sslmode=require"

    engine = create_engine(DATABASE_URL)
    return engine


def save_to_supabase(df, table_name, engine, if_exists="replace"):
    """
    Menyimpan DataFrame ke tabel Supabase.

    Parameters:
        df         : pandas DataFrame
        table_name : nama tabel di Supabase
        engine     : SQLAlchemy engine
        if_exists  : 'replace' (hapus & buat ulang) atau 'append'
    """
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"[DB SAVED] Tabel '{table_name}' -> {len(df)} baris berhasil disimpan ke Supabase")


def read_from_supabase(query_or_table, engine):
    """
    Membaca data dari Supabase.

    Parameters:
        query_or_table : nama tabel (akan di-SELECT * FROM) atau query SQL lengkap
        engine         : SQLAlchemy engine

    Returns:
        pandas DataFrame
    """
    if query_or_table.strip().upper().startswith("SELECT"):
        query = query_or_table
    else:
        query = f'SELECT * FROM "{query_or_table}"'

    import pandas as pd
    df = pd.read_sql(query, engine)
    print(f"[DB READ] {len(df)} baris dimuat dari: {query_or_table}")
    return df
