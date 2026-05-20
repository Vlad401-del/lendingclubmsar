import gzip
import csv

def filter_accepted():
    print("Filtering accepted data...")
    with gzip.open("accepted_2007_to_2018Q4.csv.gz", 'rt', encoding='utf-8') as f_in, \
         open("accepted_2018.csv", 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        try:
            header = next(reader)
        except StopIteration:
            return
            
        writer.writerow(header)
        
        try:
            issue_d_idx = header.index("issue_d")
        except ValueError:
            print("issue_d column not found")
            return
            
        count = 0
        written = 0
        for row in reader:
            if len(row) > issue_d_idx:
                val = row[issue_d_idx].strip()
                if val.endswith('2018'):
                    writer.writerow(row)
                    written += 1
            
            count += 1
            if count % 100000 == 0:
                print(f"Accepted rows processed: {count} (Kept: {written})")
                
    print(f"Finished accepted data. Total kept: {written}")

def filter_rejected():
    print("Filtering rejected data...")
    with gzip.open("rejected_2007_to_2018Q4.csv.gz", 'rt', encoding='utf-8') as f_in, \
         open("rejected_2018.csv", 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        try:
            header = next(reader)
        except StopIteration:
            return
            
        writer.writerow(header)
        
        try:
            app_date_idx = header.index("Application Date")
        except ValueError:
            print("Application Date column not found")
            return
            
        count = 0
        written = 0
        for row in reader:
            if len(row) > app_date_idx:
                val = row[app_date_idx].strip()
                if val.startswith('2018'):
                    writer.writerow(row)
                    written += 1
                    
            count += 1
            if count % 500000 == 0:
                print(f"Rejected rows processed: {count} (Kept: {written})")
                
    print(f"Finished rejected data. Total kept: {written}")

if __name__ == "__main__":
    filter_accepted()
    filter_rejected()
