import pandas as pd
from scipy import spatial
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#separar por consequence
dnabert2_df =  pd.read_csv('../../runs/dnabert2/expression_dnabert2/cosine_distance_expression_dnabert2.csv')
grover_df =  pd.read_csv('../../runs/grover/expression_grover/cosine_distance_expression_grover.csv')
hyenadna_df =  pd.read_csv('../../runs/hyenadna/expression_hyenadna/cosine_distance_expression_hyenadna.csv')
alphagenome_df =  pd.read_csv('../../runs/alphagenome/expression_ag/cosine_distance_expression_ag.csv')


dfs = {
    "dnabert2": dnabert2_df,
    "grover": grover_df,
    "hyena": hyenadna_df,
    "ag": alphagenome_df,
}

X = np.column_stack([d["distance"].values for d in dfs.values()]) 
Xz = (X - X.mean(axis=0)) / X.std(axis=0)
mean_dist = X.mean(axis=1)   # (n_variants,)
mean_dist_z = Xz.mean(axis=1)

y = dnabert2_df["label"]

print("raw mean AUC:   ", roc_auc_score(y, mean_dist))
print("z-scored mean AUC:", roc_auc_score(y, mean_dist_z))
