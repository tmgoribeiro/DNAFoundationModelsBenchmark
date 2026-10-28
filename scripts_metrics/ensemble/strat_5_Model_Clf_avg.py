import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA

def load_emb(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj.select_dtypes("number").values
    if isinstance(obj, dict):  # e.g. {"embeddings": ..., "labels": ...} or {id: vec}
        # adjust the key if needed
        obj = obj["embeddings"] if "embeddings" in obj else np.stack(list(obj.values()))
    if torch.is_tensor(obj):
        obj = obj.cpu().numpy()
    return np.asarray(obj)

def load_pairs(path):
    idx, ref, alt = [], [], []
    with open(path, "rb") as f:
        while True:
            try:
                i, r, a = pickle.load(f)
            except EOFError:
                break
            idx.append(i); ref.append(r); alt.append(a)
    return np.array(idx), np.stack(ref), np.stack(alt)



def make_model(d_in, n_out):
    out = 1 if n_out == 2 else n_out
    if args.hidden == 0 or args.layers == 0:
        return nn.Linear(d_in, out)
    layers, d = [], d_in
    for _ in range(args.layers):
        layers += [nn.Linear(d, args.hidden), nn.ReLU(), nn.Dropout(args.dropout)]
        d = args.hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)
 
 
def predict_proba(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy(), logits.cpu().numpy()
 
 
def auc(y_true, probs):
    if n_classes == 2:
        return roc_auc_score(y_true, probs)
    return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

def training_loop(X_tr, y_tr, X_te, y_te, fold, name):
    model = make_model(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() 

    n = len(X_tr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        running, steps = torch.zeros((), device=device), 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            out = model(X_tr[idx])
            out = out.squeeze(1) if n_classes == 2 else out
            loss = loss_fn(out, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.detach()
            steps += 1
        #print(f"    Epoch {epoch:3d} | train loss = {running.item() / steps:.4f}")

    # validation loss on the held-out fold
    model.eval()
    with torch.no_grad():
        val_out = model(X_te)
        val_out = val_out.squeeze(1) if n_classes == 2 else val_out
        val_loss = loss_fn(val_out, y_te).item()

    probs, logits = predict_proba(model, X_te)
    fold_auc = auc(y_te.cpu().numpy(), probs)
    print(f"Model {name}, fold {fold}: AUC = {fold_auc:.4f} | val loss = {val_loss:.4f}\n")
    return probs, logits, fold_auc


p = argparse.ArgumentParser()
p.add_argument("--folds", type=int, default=5)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--hidden", type=int, default=256, help="0 = linear classifier")
p.add_argument("--layers", type=int, default=1)
p.add_argument("--dropout", type=int, default=0.1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--random", action="store_true", help="skip training: randomly initialised model")
args = p.parse_args()
 
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
 
# ---- load data --
_, dnabert2_ref, dnabert2_alt = load_pairs("../../runs/dnabert2/expression_dnabert2/embeddings_expression_dnabert2.pkl")
_, grover_ref, grover_alt = load_pairs("../../runs/grover/expression_grover/embeddings_expression_grover.pkl")
_, hyenadna_ref, hyenadna_alt = load_pairs("../../runs/hyenadna/expression_hyenadna/embeddings_expression_hyenadna.pkl")
_, ag_ref, ag_alt = load_pairs("../../runs/alphagenome/expression_ag/embeddings_expression_ag.pkl")

dnabert2_X= (dnabert2_ref-dnabert2_alt).astype(np.float32)
grover_X= (grover_ref-grover_alt).astype(np.float32)
hyenadna_X= (hyenadna_ref-hyenadna_alt).astype(np.float32)
ag_X= (ag_ref-ag_alt).astype(np.float32)

feats = {
    "dnabert2": dnabert2_X,
    "grover": grover_X,
    "hyena": hyenadna_X,
    "ag": ag_X,
}

label_df = pd.read_csv("../../data/variant_effects/variant_effects_expression.bed", sep="\t")
LABEL_COL = "label"  # adjust to the label column name in the .bed
y_raw = label_df[LABEL_COL].values

classes, y = np.unique(y_raw, return_inverse=True)
n_classes = len(classes)

 

probabilities = np.zeros((len(y), len(feats.keys())))
logits = np.zeros((len(y), len(feats.keys())))

skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
for fold, (tr, te) in enumerate(skf.split(dnabert2_X, y), 1):

    y_dtype = torch.float32 if n_classes == 2 else torch.long
    y_tr = torch.tensor(y[tr], dtype=y_dtype, device=device)
    y_te = torch.tensor(y[te], dtype=y_dtype, device=device)

    for m, (name, Xm) in enumerate(feats.items()):

        scaler = StandardScaler().fit(Xm[tr])
        X_tr = torch.tensor(scaler.transform(Xm[tr]), dtype=torch.float32, device=device)
        X_te = torch.tensor(scaler.transform(Xm[te]), dtype=torch.float32, device=device)

        probs, logs, _ = training_loop(X_tr, y_tr, X_te, y_te, fold, name)
        probabilities[te, m] = probs
        logits[te, m] = logs

mean_probs = np.mean(probabilities, axis=1)
mean_logs = np.mean(logits, axis=1)   

final_prob_score = roc_auc_score(label_df['label'], mean_probs)
final_log_score = roc_auc_score(label_df['label'], mean_logs)

print(f"Final probability AUC: {final_prob_score}")
print(f"Final logits AUC: {final_log_score}")



