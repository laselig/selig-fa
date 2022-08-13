import pandas as pd
import os
print(os.cpu_count())

df = pd.read_parquet("/home/lselig/selig-fa/finance/.data/evs/A.parquet")
print(df.head())