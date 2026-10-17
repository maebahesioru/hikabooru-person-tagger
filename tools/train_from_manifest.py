#!/usr/bin/env python3
"""学習スクリプト: SigLIP2 vision tower + LoRA でマルチラベル分類器を作る。

manifest.jsonl の各行: {"path": "画像パス", "labels": ["hikakin", ...]}
(1枚が複数クラスを持ってよい / ラベルなし = ネガティブ)

Usage:
    python train_from_manifest.py --manifest manifest.jsonl --out out/
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score

MODEL_ID = "google/siglip2-base-patch16-384"


class DS(Dataset):
    def __init__(self, rows, proc, classes):
        self.rows, self.proc, self.classes = rows, proc, classes
        self.c2i = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = Image.open(r["path"]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (384, 384), (128, 128, 128))
        pv = self.proc(images=im, return_tensors="pt")["pixel_values"][0]
        y = np.zeros(len(self.classes), dtype=np.float32)
        for c in r.get("labels") or []:
            if c in self.c2i:
                y[self.c2i[c]] = 1
        return pv, torch.from_numpy(y)


def evaluate(net, dl, classes):
    net.eval()
    L, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        for pv, y in dl:
            L.append(net(pv.cuda()).float().cpu().numpy())
            Y.append(y.numpy())
    L, Y = np.concatenate(L), np.concatenate(Y)
    S = 1 / (1 + np.exp(-L))
    res = {}
    for i, c in enumerate(classes):
        if Y[:, i].sum() == 0:
            continue
        auc = roc_auc_score(Y[:, i], S[:, i])
        res[c] = {"auc": round(float(auc), 4), "n_pos": int(Y[:, i].sum())}
        print(f"  {c}: AUC={auc:.3f} (n={int(Y[:, i].sum())})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()

    rows = []
    for l in open(args.manifest, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if Path(r["path"]).exists():
            rows.append(r)
    classes = sorted({c for r in rows for c in (r.get("labels") or [])})
    print(f"rows={len(rows)} classes={len(classes)}", flush=True)

    rng = random.Random(42)
    rng.shuffle(rows)
    cut = int(len(rows) * (1 - args.test_ratio))
    train_rows, test_rows = rows[:cut], rows[cut:]

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID)
    vision = base.vision_model
    del base
    for p in vision.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    vision = get_peft_model(vision, cfg)
    vision.print_trainable_parameters()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision
            self.head = nn.Linear(768, len(classes))

        def forward(self, pv):
            out = self.vision(pixel_values=pv)
            pool = getattr(out, "pooler_output", None)
            if pool is None:
                pool = out.last_hidden_state.mean(dim=1)
            return self.head(pool.float())

    net = Net().cuda()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    dl_tr = DataLoader(DS(train_rows, proc, classes), batch_size=args.batch,
                       shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(DS(test_rows, proc, classes), batch_size=16,
                       shuffle=False, num_workers=4, pin_memory=True)
    bce = nn.BCEWithLogitsLoss()
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    # NOTE: GradScaler を無効にすると fp16 で勾配が壊れ発散する(実測)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        net.train()
        losses = []
        for pv, y in dl_tr:
            pv, y = pv.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(pv)
                loss = bce(logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"ep{ep+1} step {step}/{steps} loss={np.mean(losses[-100:]):.4f} "
                      f"gn={float(gn):.2f} {step/el:.2f} steps/s", flush=True)
        print(f"epoch {ep+1} done, loss={np.mean(losses):.4f}", flush=True)
        evaluate(net, dl_te, classes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net.vision.save_pretrained(out / "lora_adapter")
    torch.save({"head": net.head.state_dict(), "classes": classes}, out / "lora_head.pt")
    print("SAVED ->", out)


if __name__ == "__main__":
    main()
