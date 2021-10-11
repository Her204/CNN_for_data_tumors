import pandas as pd
import os

df = pd.read_csv("preprocessed_data_modified.csv")

a = df.copy()

a["File"] = a.File.apply(lambda x: x.split("/")[-1])

a["File"] = a.File.apply(lambda x: x.split("_")[0])

print(a.File.value_counts(),
    df.File.apply(
        lambda x:x.split("/")[3]
            ).value_counts())
