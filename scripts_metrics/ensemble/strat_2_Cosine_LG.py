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

label_df = pd.read_csv("../../data/variant_effects/variant_effects_expression.bed", sep="\t")
LABEL_COL = "label"  # adjust to the label column name in the .bed
y_raw = label_df[LABEL_COL].values
classes, y = np.unique(y_raw, return_inverse=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stack_aucs = []
for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=1)).fit(X[tr], y[tr])
    scores = meta.decision_function(X[te])   # the stacker's logit, (n_te,)
    stack_aucs.append(roc_auc_score(y[te], scores))
    print("weights:", meta[-1].coef_.round(3), "bias:", meta[-1].intercept_.round(3))

print(f"stacker: {np.mean(stack_aucs):.4f} ± {np.std(stack_aucs):.4f}")