import pandas as pd

def inspect_date():
    df_acc = pd.read_csv("accepted_2007_to_2018Q4.csv.gz", usecols=['issue_d'], nrows=5)
    print("Accepted issue_d:", df_acc['issue_d'].tolist())
    
    df_rej = pd.read_csv("rejected_2007_to_2018Q4.csv.gz", usecols=['Application Date'], nrows=5)
    print("Rejected Application Date:", df_rej['Application Date'].tolist())

if __name__ == "__main__":
    inspect_date()
