import pandas as pd
import locale

def fetch_and_format():
    print("Mendownload data dari FRED...")
    # Fetch data
    fedfunds = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS', parse_dates=['observation_date'], index_col='observation_date')
    cpi = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL', parse_dates=['observation_date'], index_col='observation_date')
    drcl = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DRCLACBS', parse_dates=['observation_date'], index_col='observation_date')

    # Convert to numeric, errors='coerce' to handle any missing dots
    fedfunds['FEDFUNDS'] = pd.to_numeric(fedfunds['FEDFUNDS'], errors='coerce')
    cpi['CPIAUCSL'] = pd.to_numeric(cpi['CPIAUCSL'], errors='coerce')
    drcl['DRCLACBS'] = pd.to_numeric(drcl['DRCLACBS'], errors='coerce')

    # Calculate YoY Inflation
    cpi['Inflation'] = cpi['CPIAUCSL'].pct_change(12) * 100

    # Resample DRCLACBS to monthly and forward fill
    drcl_monthly = drcl.resample('MS').ffill()

    # Combine
    df = pd.concat([fedfunds, cpi[['Inflation']], drcl_monthly], axis=1)

    # Filter 2014 to 2018
    df = df.loc['2014-01-01':'2018-12-31'].copy()
    
    # Fill any remaining NaNs in DRCLACBS (e.g. if the first months of 2014 didn't have a value from Q4 2013, though it should)
    df['DRCLACBS'] = df['DRCLACBS'].bfill().ffill()

    # Sort descending like the Indonesian file
    df = df.sort_index(ascending=False)

    # Formatting functions
    def format_indonesian_date(ts):
        months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        return f"{months[ts.month - 1]} {ts.year}"

    def format_percent(val):
        if pd.isna(val):
            return ""
        # format 2 decimal places, replace dot with comma, add %
        return f"{val:.2f}".replace('.', ',') + "%"

    df['Bulan dan Tahun'] = df.index.map(format_indonesian_date)
    df['Fed-Funds-Rate'] = df['FEDFUNDS'].apply(format_percent)
    df['Inflasi'] = df['Inflation'].apply(format_percent)
    df['Delinquency_Rate'] = df['DRCLACBS'].apply(format_percent)

    # Select and rename columns
    final_df = df[['Bulan dan Tahun', 'Fed-Funds-Rate', 'Inflasi', 'Delinquency_Rate']]

    # Save to CSV
    # We must quote all fields like the example. Pandas does not quote everything by default.
    import csv
    final_df.to_csv('TrenEkonomiAS_2014_2018_EA.csv', index=False, quoting=csv.QUOTE_ALL)
    print("Selesai! File disimpan sebagai TrenEkonomiAS_2014_2018_EA.csv")

if __name__ == '__main__':
    fetch_and_format()
