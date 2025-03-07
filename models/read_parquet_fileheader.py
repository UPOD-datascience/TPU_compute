import pandas as pd 
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str)
args = argparser.parse_args()
df = pd.read_parquet(args.filename, engine='pyarrow')
print(df.columns)

# total_text