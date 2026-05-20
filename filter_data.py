import pandas as pd
import gc

def process_accepted():
    print("Processing accepted data...")
    input_file = 'accepted_2007_to_2018Q4.csv.gz'
    output_file = 'accepted_2014_2018_sampled.csv'
    chunksize = 200000
    chunks = []
    
    try:
        # Read the CSV file in chunks to handle large file size
        for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize, low_memory=False)):
            # Extract year safely. 'issue_d' format is typically 'MMM-YYYY' (e.g., 'Dec-2015')
            year_str = chunk['issue_d'].str.extract(r'(\d{4})')[0]
            year_num = pd.to_numeric(year_str, errors='coerce')
            
            # Filter data to keep only years from 2014 to 2018
            filtered_chunk = chunk[(year_num >= 2014) & (year_num <= 2018)]
            
            # If there's data left after filtering, take a 5% random sample
            if not filtered_chunk.empty:
                sampled_chunk = filtered_chunk.sample(frac=0.05, random_state=42)
                chunks.append(sampled_chunk)
            
            print(f"  Processed accepted chunk {i+1}...")
            
        # Concatenate all sampled chunks into a single DataFrame and save
        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"[SUCCESS] Accepted data saved to {output_file} with {len(final_df)} rows.\n")
        else:
            print("[ERROR] No data matched the criteria for accepted loans.\n")
            
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}. Please ensure it is in the current directory.\n")
        
    del chunks
    gc.collect()

def process_rejected():
    print("Processing rejected data...")
    input_file = 'rejected_2007_to_2018Q4.csv.gz'
    output_file = 'rejected_2014_2018_sampled.csv'
    chunksize = 500000
    chunks = []
    
    try:
        # Read the CSV file in chunks to handle large file size
        for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize, low_memory=False)):
            # Extract year safely. 'Application Date' format is typically 'YYYY-MM-DD'
            year_str = chunk['Application Date'].str.extract(r'^(\d{4})')[0]
            year_num = pd.to_numeric(year_str, errors='coerce')
            
            # Filter data to keep only years from 2014 to 2018
            filtered_chunk = chunk[(year_num >= 2014) & (year_num <= 2018)]
            
            # If there's data left after filtering, take a 5% random sample
            if not filtered_chunk.empty:
                sampled_chunk = filtered_chunk.sample(frac=0.05, random_state=42)
                chunks.append(sampled_chunk)
                
            print(f"  Processed rejected chunk {i+1}...")
            
        # Concatenate all sampled chunks into a single DataFrame and save
        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"[SUCCESS] Rejected data saved to {output_file} with {len(final_df)} rows.\n")
        else:
            print("[ERROR] No data matched the criteria for rejected loans.\n")
            
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}. Please ensure it is in the current directory.\n")
        
    del chunks
    gc.collect()

if __name__ == "__main__":
    print("Starting Pre-processing...\n")
    process_accepted()
    process_rejected()
    print("[DONE] Pre-processing completed!")
