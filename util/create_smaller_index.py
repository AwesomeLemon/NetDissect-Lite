import pandas as pd
import random
p = 0.1  # 10% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
input_path = 'dataset/broden1_224/index.csv'
output_path = 'dataset/broden1_224/index_sm.csv'
df = pd.read_csv(
         input_path,
         header=0,
         skiprows=lambda i: i>0 and random.random() > p
)
df2 = df[df['texture'].isnull()]
df2.to_csv(output_path, index=False)