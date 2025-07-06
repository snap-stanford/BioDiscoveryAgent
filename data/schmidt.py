import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from screen import ScreenData
import scanpy as sc

out_file = "ground_truth_IFNG.csv"
dir_path = "./datas"

### -------

d1_file_path = os.path.join(dir_path, "schmidt_ifng_d1.gene_summary.txt")
df = pd.read_csv(d1_file_path, sep="\t", index_col="id")
d2_file_path = os.path.join(dir_path, "schmidt_ifng_d2.gene_summary.txt")
df_d2 = pd.read_csv(d2_file_path, sep="\t", index_col="id")

df = pd.concat([df, df_d2])
group_by_row_index = df.groupby(df.index)
df = group_by_row_index.mean()

gene_names = df.index.values.tolist()
name_converter = HGNCNames('/dfs/user/yhr/genedisco/genedisco/datasets/screens/')
gene_names = name_converter.update_outdated_gene_names(gene_names)
df.index = gene_names

gene_names, data = df.index.values.tolist(), df[['pos|lfc']].values.astype(np.float32)

# Merge duplicate indices by averaging
df = df.groupby(df.index).mean()
df['pos|lfc'] = df['pos|lfc'].values.astype(np.float32)
df = df.loc[:,['pos|lfc']]
df = df.rename(columns = {'pos|lfc':'log-fold-change'})
df = df.reset_index().rename(columns={'index':'Gene'})

df.to_csv(out_file, index=0)

file_path = out_file
id_col = 'Gene'
val_col = 'log-fold-change'
bio_taskname = 'IFNG'

screendata = ScreenData(file_path=file_path, id_col=id_col,
       val_col=val_col, bio_taskname=bio_taskname, save=True)
screendata.identify_hits(type_='gaussian', save=True)
