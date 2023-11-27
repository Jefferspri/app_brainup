import os
import pandas as pd
from datetime import datetime
from natsort import os_sorted

file = "validity/eeg/eeg_16_Jefferson_2.csv"

df_eeg = pd.read_csv(file, sep=',')
df_eeg['time'] = df_eeg['time'].map(lambda x: datetime.fromtimestamp(x))
df_eeg.to_csv(file, index=False)