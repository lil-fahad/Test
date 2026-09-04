from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
MODELS = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
EPOCHS = 5

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def emit(payload: dict[str, object]) -> None:
    print("NOVATRAIN_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)


def model_factory(name: str) -> nn.Module:
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 10)
        return m
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 10)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 10)
        return m
    raise ValueError(name)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = ce(logits, y)
            loss_sum += float(loss) * y.numel()
            correct += int((logits.argmax(1) == y).sum())
            total += y.numel()
    return {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total)}


def latest_checkpoint(checkpoint_dir: Path, model_name: str) -> Path | None:
    p = checkpoint_dir / f"{model_name}_last.pt"
    return p if p.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This production training job requires CUDA")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    out = Path(".novatrain/output")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(os.environ.get("NOVATRAIN_CHECKPOINT_DIR", ".novatrain/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(os.environ.get("NOVATRAIN_DATA_CACHE", ".novatrain/datasets")) / "cifar10"
    data_root.mkdir(parents=True, exist_ok=True)

    scale = float(os.environ.get("NOVATRAIN_BATCH_SCALE", "1"))
    batch_size = max(16, int(args.batch_size * max(0.0625, min(1.0, scale))))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    emit({"phase": "dataset", "message": "loading_cifar10", "progress": 0.01})
    train_ds = datasets.CIFAR10(str(data_root), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(str(data_root), train=False, download=True, transform=test_tf)
    train_ds = Subset(train_ds, list(range(30000)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(128, min(1024, batch_size * 2)),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    suite: dict[str, object] = {}
    best_overall_name = ""
    best_overall_accuracy = -1.0
    best_overall_path: Path | None = None

    for model_index, name in enumerate(MODELS, start=1):
        model = model_factory(name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda")
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        history: list[dict[str, float | int]] = []
        best = -1.0
        started = time.time()
        start_epoch = 1

        resume_path = latest_checkpoint(checkpoint_dir, name)
        if args.resume and Path(args.resume).is_file():
            candidate = Path(args.resume)
            try:
                probe = torch.load(candidate, map_location="cpu", weights_only=False)
                if probe.get("model") == name:
                    resume_path = candidate
            except Exception:
                pass
        if resume_path is not None:
            try:
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                if ckpt.get("model") == name:
                    model.load_state_dict(ckpt["state_dict"])
                    if "optimizer" in ckpt:
                        opt.load_state_dict(ckpt["optimizer"])
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    best = float(ckpt.get("best_accuracy", ckpt.get("metrics", {}).get("val_accuracy", -1.0)))
                    prior_history = ckpt.get("history", [])
                    if isinstance(prior_history, list):
                        history = prior_history
                    for _ in range(max(0, start_epoch - 1)):
                        sched.step()
                    emit({
                        "phase": "resume",
                        "model": name,
                        "model_index": model_index,
                        "models_total": len(MODELS),
                        "epoch": start_epoch - 1,
                        "epochs_total": EPOCHS,
                        "message": "checkpoint_loaded",
                    })
            except Exception as exc:
                emit({"phase": "resume", "model": name, "message": f"checkpoint_ignored:{type(exc).__name__}"})

        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            total = correct = 0
            loss_sum = 0.0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = ce(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                n = y.numel()
                total += n
                loss_sum += float(loss) * n
                correct += int((logits.argmax(1) == y).sum())

            sched.step()
            val = evaluate(model, test_loader, device)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, total),
                "train_accuracy": correct / max(1, total),
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "lr": opt.param_groups[0]["lr"],
            }
            history.append(row)
            best = max(best, float(val["accuracy"]))
            ckpt = {
                "model": name,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "metrics": row,
                "history": history,
                "best_accuracy": best,
                "classes": train_ds.dataset.classes,
                "batch_size": batch_size,
            }
            stable_last = checkpoint_dir / f"{name}_last.pt"
            torch.save(ckpt, stable_last)
            shutil.copy2(stable_last, out / f"{name}_last.pt")
            if float(val["accuracy"]) >= best:
                stable_best = checkpoint_dir / f"{name}_best.pt"
                torch.save(ckpt, stable_best)
                shutil.copy2(stable_best, out / f"{name}_best.pt")

            progress = ((model_index - 1) * EPOCHS + epoch) / (len(MODELS) * EPOCHS)
            emit({
                "phase": "training",
                "model": name,
                "model_index": model_index,
                "models_total": len(MODELS),
                "epoch": epoch,
                "epochs_total": EPOCHS,
                "train_loss": row["train_loss"],
                "train_accuracy": row["train_accuracy"],
                "val_loss": row["val_loss"],
                "val_accuracy": row["val_accuracy"],
                "best_accuracy": best,
                "batch_size": batch_size,
                "progress": progress,
            })

        suite[name] = {
            "best_accuracy": best,
            "history": history,
            "seconds": round(time.time() - started, 2),
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": batch_size,
        }
        candidate_best = checkpoint_dir / f"{name}_best.pt"
        if best > best_overall_accuracy and candidate_best.is_file():
            best_overall_accuracy = best
            best_overall_name = name
            best_overall_path = candidate_best
        del model
        torch.cuda.empty_cache()

    if best_overall_path is not None:
        shutil.copy2(best_overall_path, out / "best_overall.pt")

    result = {
        "ok": True,
        "request_id": os.environ.get("NOVATRAIN_REQUEST_ID"),
        "repository": os.environ.get("NOVATRAIN_REPOSITORY"),
        "commit": os.environ.get("NOVATRAIN_COMMIT_SHA"),
        "worker_version": os.environ.get("NOVATRAIN_WORKER_VERSION"),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "best_model": best_overall_name,
        "best_accuracy": best_overall_accuracy,
        "suite": suite,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    emit({
        "phase": "completed",
        "progress": 1.0,
        "best_model": best_overall_name,
        "best_accuracy": best_overall_accuracy,
    })
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
