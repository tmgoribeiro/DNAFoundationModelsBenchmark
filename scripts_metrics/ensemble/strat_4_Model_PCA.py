import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import time
from datetime import datetime
from pathlib import Path
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
        logits = model(X_t)
        if n_classes == 2:
            return torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return torch.softmax(logits, dim=1).cpu().numpy()
 
 
def auc(y_true, probs):
    if n_classes == 2:
        return roc_auc_score(y_true, probs)
    return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")


def aupr(y_true, probs):
    if n_classes == 2:
        return average_precision_score(y_true, probs)
    return average_precision_score(label_binarize(y_true, classes=np.arange(n_classes)), probs, average="macro")


def now():
    # the GPU runs asynchronously, so wait for it before reading the clock
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def peak_ram_gb():
    try:
        import resource  # not available on Windows
    except ImportError:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
 


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
p.add_argument("--models", nargs="+", choices=["dnabert2", "grover", "hyena", "ag"], default=["hyena", "ag"], help="models to concatenate")
p.add_argument("--n_components", type=int, default=128, help="PCA dimensions kept for each model")
args = p.parse_args()
 
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
 
start = now()

# ---- load data --
paths = {
    "dnabert2": "../../runs/dnabert2/expression_dnabert2/embeddings_expression_dnabert2.pkl",
    "grover": "../../runs/grover/expression_grover/embeddings_expression_grover.pkl",
    "hyena": "../../runs/hyenadna/expression_hyenadna/embeddings_expression_hyenadna.pkl",
    "ag": "../../runs/alphagenome/expression_ag/embeddings_expression_ag.pkl",
}

feats = {}
for name in args.models:
    _, ref, alt = load_pairs(paths[name])
    feats[name] = (ref - alt).astype(np.float32)

label_df = pd.read_csv("../../data/variant_effects/variant_effects_expression.bed", sep="\t")
LABEL_COL = "label"  # adjust to the label column name in the .bed
y_raw = label_df[LABEL_COL].values

classes, y = np.unique(y_raw, return_inverse=True)
n_classes = len(classes)
#print(f"X: {X.shape} | classes: {n_classes} {dict(zip(classes, np.bincount(y)))} | device: {device}")
 
 
load_time = now() - start

skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
aucs, auprs, pca_times, train_times, predict_times = [], [], [], [], []
 
for fold, (tr, te) in enumerate(skf.split(feats[args.models[0]], y), 1):
    fold_start = now()

    tr_blocks, te_blocks = [], []
    for Xm in feats.values():
        sc = StandardScaler().fit(Xm[tr])
        Xtr_s, Xte_s = sc.transform(Xm[tr]), sc.transform(Xm[te])
        pca = PCA(n_components=args.n_components, svd_solver="randomized", random_state=args.seed).fit(Xtr_s)
        tr_blocks.append(pca.transform(Xtr_s))
        te_blocks.append(pca.transform(Xte_s))

    X_tr = torch.from_numpy(np.concatenate(tr_blocks, axis=1)).float().to(device)
    X_te = torch.from_numpy(np.concatenate(te_blocks, axis=1)).float().to(device)

    pca_times.append(now() - fold_start)
    print(f"Fold {fold} | models: {args.models} | X: {tuple(X_tr.shape)}")
    y_dtype = torch.float32 if n_classes == 2 else torch.long
    y_tr = torch.tensor(y[tr], dtype=y_dtype, device=device)
    y_te = torch.tensor(y[te], dtype=y_dtype, device=device)
    loss_fn = nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()

    model = make_model(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_start = now()
    if not args.random:
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
            print(f"Fold {fold} | epoch {epoch:3d} | train loss = {running.item() / steps:.4f}")

    train_times.append(now() - train_start)

    # validation loss on the held-out fold
    model.eval()
    with torch.no_grad():
        val_out = model(X_te)
        val_out = val_out.squeeze(1) if n_classes == 2 else val_out
        val_loss = loss_fn(val_out, y_te).item()

    predict_start = now()
    probs = predict_proba(model, X_te)
    predict_times.append(now() - predict_start)
    fold_auc = auc(y[te], probs)
    fold_aupr = aupr(y[te], probs)
    aucs.append(fold_auc)
    auprs.append(fold_aupr)
    print(f"Fold {fold}: AUC = {fold_auc:.4f} | AUPR = {fold_aupr:.4f} | val loss = {val_loss:.4f} | "
          f"pca {pca_times[-1]:.1f}s | train {train_times[-1]:.1f}s | predict {predict_times[-1]:.3f}s\n")
 
elapsed = now() - start
chance_aupr = y.mean() if n_classes == 2 else 1 / n_classes
peak_gpu = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else float("nan")

print(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f} | AUPR by chance: {chance_aupr:.4f} | positive class: {classes[-1]}")
print(f"Time: total {elapsed:.1f}s | load {load_time:.1f}s | per fold: pca {np.mean(pca_times):.1f}s, train {np.mean(train_times):.1f}s, predict {np.mean(predict_times):.3f}s")
print(f"Peak memory: GPU {peak_gpu:.2f} GB | RAM {peak_ram_gb():.2f} GB | device: {device}")

log_dir = Path("../../runs/ensemble")
log_dir.mkdir(parents=True, exist_ok=True)
with open(log_dir / "log.txt", "a") as log_f:
    log_f.write(
        f"{datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"script={Path(__file__).stem} | "
        f"models={'+'.join(args.models)} | "
        f"n_components={args.n_components} | "
                f"X_dim={X_tr.shape[1]} | "
        f"n_variants={len(y)} | "
        f"AUC={np.mean(aucs):.4f}+-{np.std(aucs):.4f} | "
        f"AUPR={np.mean(auprs):.4f}+-{np.std(auprs):.4f} | "
        f"chance_AUPR={chance_aupr:.4f} | "
        f"elapsed={elapsed:.1f}s | "
        f"load={load_time:.1f}s | "
        f"pca_per_fold={np.mean(pca_times):.1f}s | "
        f"train_per_fold={np.mean(train_times):.1f}s | "
        f"predict_per_fold={np.mean(predict_times):.3f}s | "
        f"peak_gpu={peak_gpu:.2f}GB | "
        f"peak_ram={peak_ram_gb():.2f}GB | "
        f"device={device}\n"
    )
