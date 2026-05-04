import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dataset2d import SegmentationDataset2D
    from unet2d import UNet2D
    from metrics import threshold_logits, dice_coeff, iou_score
except ImportError:
    from src.dataset2d import SegmentationDataset2D
    from src.unet2d import UNet2D
    from src.metrics import threshold_logits, dice_coeff, iou_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(data_dir: Path, batch_size: int, shuffle: bool, num_workers: int, seed: int, device: torch.device):
    ds = SegmentationDataset2D(str(data_dir), augment=shuffle, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    ), ds


def estimate_pos_weight(loader: DataLoader, device: torch.device, max_batches: int = 10) -> float:
    pos = 0.0
    neg = 0.0
    seen = 0

    for _, masks in loader:
        masks = masks.to(device)
        pos += masks.sum().item()
        neg += masks.numel() - masks.sum().item()
        seen += 1
        if seen >= max_batches:
            break

    if pos + neg == 0:
        return 1.0

    pw = neg / (pos + 1e-6)
    return float(min(pw, 100.0))


def make_scaler(device: torch.device, use_amp: bool):
    if not (device.type == "cuda" and use_amp):
        return None

    try:
        return GradScaler("cuda")
    except TypeError:
        try:
            return GradScaler()
        except TypeError:
            return None


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler, grad_clip):
    model.train()
    running_loss = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        running_loss += loss.item()
        count += 1

    return running_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, thr):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = threshold_logits(logits, thr=thr)
        dice = dice_coeff(preds, masks)
        iou = iou_score(preds, masks)

        running_loss += loss.item()
        running_dice += float(dice)
        running_iou += float(iou)
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    return (
        running_loss / count,
        running_dice / count,
        running_iou / count,
    )


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = [
        "experiment",
        "best_epoch",
        "best_val_dice",
        "best_val_iou",
        "test_target",
        "test_loss",
        "test_dice",
        "test_iou",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_experiment(
    experiment_name: str,
    train_root: Path,
    test_roots: dict,
    args,
    device: torch.device,
    results_dir: Path,
):
    print("\n" + "=" * 80)
    print(f"Running experiment: {experiment_name}")
    print("=" * 80)

    train_loader, train_ds = make_loader(
        train_root / "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    val_loader, val_ds = make_loader(
        train_root / "val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    print(f"train samples: {len(train_ds)}")
    print(f"val samples:   {len(val_ds)}")

    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)

    pos_weight = estimate_pos_weight(train_loader, device=device)
    print(f"pos_weight:    {pos_weight:.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scaler = make_scaler(device, args.amp)
    use_amp = (device.type == "cuda" and args.amp and scaler is not None)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    best_val_dice = -1.0
    best_val_iou = 0.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )

        val_loss, val_dice, val_iou = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            thr=args.thr,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
        )

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    model.load_state_dict(best_state)

    exp_dir = results_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "history": history,
        },
        exp_dir / "best_checkpoint.pt",
    )

    test_results = {}
    summary_rows = []

    for test_name, test_root in test_roots.items():
        test_loader, test_ds = make_loader(
            test_root / "test",
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
        )

        test_loss, test_dice, test_iou = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            thr=args.thr,
        )

        print(
            f"test on {test_name:>10} | "
            f"count={len(test_ds)} | "
            f"loss={test_loss:.4f} | "
            f"dice={test_dice:.4f} | "
            f"iou={test_iou:.4f}"
        )

        test_results[test_name] = {
            "count": len(test_ds),
            "loss": test_loss,
            "dice": test_dice,
            "iou": test_iou,
        }

        summary_rows.append(
            {
                "experiment": experiment_name,
                "best_epoch": best_epoch,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "test_target": test_name,
                "test_loss": test_loss,
                "test_dice": test_dice,
                "test_iou": test_iou,
            }
        )

    save_json(
        exp_dir / "summary.json",
        {
            "experiment": experiment_name,
            "train_root": str(train_root),
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "history": history,
            "test_results": test_results,
        },
    )

    return {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "test_results": test_results,
        "summary_rows": summary_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Run cross-domain segmentation benchmark.")
    parser.add_argument("--benchmark-root", type=str, default=str(ROOT / "experiments" / "cross_domain_data"))
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results" / "cross_domain_benchmark"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    benchmark_root = Path(args.benchmark_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    approach1_root = benchmark_root / "approach1"
    approach2_root = benchmark_root / "approach2"

    if not approach1_root.exists():
        raise FileNotFoundError(f"Missing benchmark folder: {approach1_root}")
    if not approach2_root.exists():
        raise FileNotFoundError(f"Missing benchmark folder: {approach2_root}")

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"benchmark root: {benchmark_root}")
    print(f"results dir:    {results_dir}")

    all_summary_rows = []
    all_results = {}

    exp1 = run_experiment(
        experiment_name="train_on_approach1",
        train_root=approach1_root,
        test_roots={
            "approach1": approach1_root,
            "approach2": approach2_root,
        },
        args=args,
        device=device,
        results_dir=results_dir,
    )
    all_results[exp1["experiment"]] = {
        "best_epoch": exp1["best_epoch"],
        "best_val_dice": exp1["best_val_dice"],
        "best_val_iou": exp1["best_val_iou"],
        "test_results": exp1["test_results"],
    }
    all_summary_rows.extend(exp1["summary_rows"])

    exp2 = run_experiment(
        experiment_name="train_on_approach2",
        train_root=approach2_root,
        test_roots={
            "approach1": approach1_root,
            "approach2": approach2_root,
        },
        args=args,
        device=device,
        results_dir=results_dir,
    )
    all_results[exp2["experiment"]] = {
        "best_epoch": exp2["best_epoch"],
        "best_val_dice": exp2["best_val_dice"],
        "best_val_iou": exp2["best_val_iou"],
        "test_results": exp2["test_results"],
    }
    all_summary_rows.extend(exp2["summary_rows"])

    summary_json = results_dir / "benchmark_summary.json"
    summary_csv = results_dir / "benchmark_summary.csv"

    save_json(summary_json, all_results)
    write_summary_csv(summary_csv, all_summary_rows)

    print("\nSummary JSON written to:")
    print(summary_json)
    print("Summary CSV written to:")
    print(summary_csv)

    print("\nFinal quick summary:\n")
    for exp_name, exp_data in all_results.items():
        print(exp_name)
        print(
            f"  best_epoch={exp_data['best_epoch']} | "
            f"best_val_dice={exp_data['best_val_dice']:.4f}"
        )
        for test_name, scores in exp_data["test_results"].items():
            print(
                f"  test on {test_name:>10}: "
                f"dice={scores['dice']:.4f}, iou={scores['iou']:.4f}"
            )
        print()


if __name__ == "__main__":
    main()