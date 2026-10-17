#!/usr/bin/env python3
"""hikabooru person-tagger — CLI inference.

Usage:
    python classify.py image.jpg                 # top-5 を表示
    python classify.py img1.jpg img2.png --json  # JSONで出力
    python classify.py folder/*.jpg --top 3

モデル: model/lora_adapter (SigLIP2 vision tower への LoRA) + model/lora_head.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, SiglipModel
from peft import PeftModel

REPO = Path(__file__).resolve().parent.parent
MODEL_ID = "google/siglip2-base-patch16-384"


def load_model(device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    base = SiglipModel.from_pretrained(MODEL_ID)
    vision = PeftModel.from_pretrained(base.vision_model, str(REPO / "model" / "lora_adapter"))
    vision = vision.to(dev).eval()
    ck = torch.load(REPO / "model" / "lora_head.pt", map_location="cpu", weights_only=True)
    head = nn.Linear(768, len(ck["classes"]))
    head.load_state_dict(ck["head"])
    head = head.to(dev).eval()
    return proc, vision, head, ck["classes"], dev


@torch.no_grad()
def classify_images(paths, proc, vision, head, classes, dev, batch=16):
    """画像パスのリスト -> [{class: prob, ...}, ...]"""
    all_probs = []
    for i in range(0, len(paths), batch):
        ims = []
        for p in paths[i:i + batch]:
            try:
                ims.append(Image.open(p).convert("RGB"))
            except Exception:
                ims.append(Image.new("RGB", (384, 384), (128, 128, 128)))
        pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            out = vision(pixel_values=pv)
            pool = getattr(out, "pooler_output", None)
            if pool is None:
                pool = out.last_hidden_state.mean(dim=1)
            probs = torch.sigmoid(head(pool.float())).float().cpu().numpy()
        all_probs.extend(probs)
    return [{c: round(float(p), 4) for c, p in zip(classes, row)} for row in all_probs]


def main():
    ap = argparse.ArgumentParser(description="hikabooru person-tagger")
    ap.add_argument("images", nargs="+", help="画像ファイルパス")
    ap.add_argument("--top", type=int, default=5, help="上位何クラスを表示するか")
    ap.add_argument("--json", action="store_true", help="JSONで出力")
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"file not found: {missing}", file=sys.stderr)
        sys.exit(1)

    proc, vision, head, classes, dev = load_model()
    results = classify_images(paths, proc, vision, head, classes, dev)

    if args.json:
        print(json.dumps(
            [{"file": str(p), "scores": r} for p, r in zip(paths, results)],
            ensure_ascii=False, indent=1))
        return

    for p, r in zip(paths, results):
        print(str(p))
        for c, s in sorted(r.items(), key=lambda x: -x[1])[:args.top]:
            print(f"  {c:<10} {s:.4f}")
        print()


if __name__ == "__main__":
    main()
