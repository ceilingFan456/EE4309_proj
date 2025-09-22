# -*- coding: utf-8 -*-
import argparse
import glob
import os
import torch
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

from src.models import build_model, AVAILABLE_MODELS
from src.datasets.voc import VOC_CLASSES


def load_image(path):
    return Image.open(path).convert("RGB")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)  # image or folder
    ap.add_argument("--output", type=str, default="runs/infer_vis")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default=None)
    return ap.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    saved_model = None
    if isinstance(saved_args, dict):
        saved_model = saved_args.get("model")
    model_name = (args.model or saved_model or "vit").lower()
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {AVAILABLE_MODELS}")
    model = build_model(model_name, num_classes=21)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    paths = []
    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*")))
    else:
        paths = [args.input]

    os.makedirs(args.output, exist_ok=True)
    for p in paths:
        try:
            img_pil = load_image(p)
        except Exception:
            continue
        img = F.to_tensor(img_pil).to(device)
        with torch.no_grad():
            pred = model([img])[0]
        keep = pred["scores"] >= args.score_thr
        boxes = pred["boxes"][keep].cpu()
        labels = pred["labels"][keep].cpu()
        scores = pred["scores"][keep].cpu()

        # Handle case when no detections meet the threshold
        if len(boxes) > 0:
            names = [VOC_CLASSES[i - 1] for i in labels.tolist()]
            lbl_txt = [f"{n}:{s:.2f}" for n, s in zip(names, scores.tolist())]
            vis = draw_bounding_boxes((img.cpu() * 255).byte(), boxes, labels=lbl_txt, width=2)
        else:
            # No detections - just convert image to visualization format
            vis = (img.cpu() * 255).byte()

        out_path = os.path.join(args.output, Path(p).stem + "_det.jpg")
        Image.fromarray(vis.permute(1, 2, 0).numpy()).save(out_path)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
