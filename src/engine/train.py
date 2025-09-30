# -*- coding: utf-8 -*-
import os, argparse, time
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm



from src.datasets.voc import VOCDataset, collate_fn
from src.utils.transforms import Compose, ToTensor, RandomHorizontalFlip
from src.models import build_model, AVAILABLE_MODELS
from src.utils.common import seed_everything, save_jsonl


def get_args():
    ap = argparse.ArgumentParser()

    # Default to using trainval for both; subset sizes control the split
    ap.add_argument("--train-set", type=str, default="trainval")
    ap.add_argument("--val-set", type=str, default="trainval")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default="vit")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--train-subset-size", type=int, default=None)
    ap.add_argument("--val-subset-size", type=int, default=None)

    return ap.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = Compose([RandomHorizontalFlip(0.5), ToTensor()])
    val_tf = Compose([ToTensor()])

    # If both use trainval dataset, split it into train and val
    if args.train_set == "trainval" and args.val_set == "trainval":
        # Load the complete trainval dataset
        full_dataset = VOCDataset(image_set="trainval", transforms=None)
        total_size = len(full_dataset)

        # Calculate split point (80% train, 20% val)
        if args.train_subset_size or args.val_subset_size:
            # Use specified subset sizes with bounds checking
            train_size = args.train_subset_size if args.train_subset_size else int(0.8 * total_size)
            val_size = args.val_subset_size if args.val_subset_size else int(0.2 * total_size)

            end_train = min(train_size, total_size)
            end_val = min(end_train + val_size, total_size)
            train_indices = range(0, end_train)
            val_indices = range(end_train, end_val)
        else:
            # Use 80/20 split of full dataset
            split_point = int(0.8 * total_size)
            train_indices = range(0, split_point)
            val_indices = range(split_point, total_size)

        # Create train and val subsets
        train_set = VOCDataset(image_set="trainval", transforms=train_tf)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = VOCDataset(image_set="trainval", transforms=val_tf)
        val_set = torch.utils.data.Subset(val_set, val_indices)
    else:
        # Use original logic (train_set and val_set from different datasets)
        base_train = VOCDataset(image_set=args.train_set, transforms=train_tf)
        if args.train_subset_size:
            end_train = min(args.train_subset_size, len(base_train))
            train_set = torch.utils.data.Subset(base_train, range(end_train))
        else:
            train_set = base_train
        base_val = VOCDataset(image_set=args.val_set, transforms=val_tf)
        if args.val_subset_size:
            end_val = min(args.val_subset_size, len(base_val))
            val_set = torch.utils.data.Subset(base_val, range(end_val))
        else:
            val_set = base_val
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Resolve default output directory after parsing to depend on the model name.
    output_dir = args.output or f"runs/{args.model}_voc07"
    args.output = str(output_dir)

    num_classes = 21  # VOC 20 classes + background
    model = build_model(args.model, num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    sched = StepLR(optim, step_size=6, gamma=0.1)

    # Only enable AMP if CUDA is available and not explicitly disabled
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=100, desc=f"train[{epoch}/{args.epochs}]")
        loss_sum = 0.0

        for images, targets in pbar:
            # ===== STUDENT TODO: Implement training step =====
            # Hint: Complete the training loop:
            # 1. Use autocast context for mixed precision if enabled
            # 2. Forward pass: get loss_dict from model(images, targets)
            # 3. Sum all losses from the loss dictionary
            # 4. Backward pass: scale losses, compute gradients, step optimizer
            # 5. Update scaler for mixed precision training
            # raise NotImplementedError("Training step not implemented")

            # 1. use autocast context for mixed precision if enabled
            with autocast(enabled=use_amp):
                ## move things onto device. 
                print("image.device:", images[0].device)

                # images = list(image.to(device) for image in images)
                # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # # 2. Forward pass: get loss_dict from model(images, targets)
                # loss_dict = model(images, targets)

                # # 3. Sum all losses from the loss dictionary
                # losses = sum(loss for loss in loss_dict.values())
            # ==================================================

            loss_sum += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.3f}")

        sched.step()
        avg_loss = loss_sum / len(train_loader)
        save_jsonl([{"epoch": epoch, "loss": avg_loss}], os.path.join(args.output, "logs.jsonl"))

        # ===== STUDENT TODO: Implement mAP evaluation =====
        # Hint: Implement validation loop to compute mAP@0.5:
        # 1. Import and initialize MeanAveragePrecision from torchmetrics
        # 2. Set model to eval mode and disable gradients
        # 3. Loop through validation data:
        #    - Move images to device
        #    - Get model predictions (no targets needed for inference)
        #    - Update metric with predictions and ground truth targets
        # 4. Compute final mAP and extract the "map" value
        # Handle exceptions gracefully and set map50 = -1.0 if evaluation fails
        try:
            raise NotImplementedError("mAP evaluation not implemented")
        except Exception as e:
            print("Eval skipped due to:", e)
            map50 = -1.0
        # ===================================================

        is_best = map50 > best_map
        best_map = max(best_map, map50)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "best_map": best_map,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.output, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output, "best.pt"))
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}  mAP@0.5={map50:.4f}  best={best_map:.4f}")


if __name__ == "__main__":
    main()
