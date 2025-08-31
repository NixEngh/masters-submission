import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.read_runs import get_best_trial, instantiate_model
from utils.train_utils import (
    train_model,
    find_best_threshold,
    evaluate_hard_dice,
    get_device,
    validate,
)
from utils.data import H5IMCDataset
from loss_functions import SoftDiceLoss


# ----------------------------
# User-editable config
# ----------------------------
RUNS = {
    "pretrained_th": "run/peripherin_transfer_th_2025-08-19_00-55-42",
    "pretrained_nf": "run/peripherin_transfer_nf_2025-08-18_10-20-50",
    "scratch":       "run/peripherin_scratch_2025-08-16_20-26-08",
}

H5_PATH = "data.h5"
PERIPHERIN_CHANNEL = 31

# Defaults if missing from metadata.json
DEFAULTS = dict(
    lr=5e-4,
    batch_size=16,
    weight_decay=1e-3,
    betas=(0.9, 0.999),
    epochs=1,
)

# Plateau behavior via the existing training loop
STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 7
LR_REDUCTION_FACTOR = 0.7
LR_COOLDOWN = 2
MIN_LR = 1e-6
ANNEAL_PATIENCE = 10**9   # effectively never anneal
UNFREEZE_PATIENCE = 0     # no encoder freeze/unfreeze


class DiceOnlyLossAdapter(nn.Module):
    # Wrap SoftDiceLoss but present (loss, components) like your training loop expects.
    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss()
        self._w = {"w_dice": 1.0, "w_cldice": 0.0, "w_bce": 0.0}

    def forward(self, outputs, labels):
        loss = self.dice(outputs, labels)
        comps = {
            "dice": loss.detach(),
            "cldice": torch.tensor(0.0, device=loss.device),
            "bce": torch.tensor(0.0, device=loss.device),
        }
        return loss, comps

    # Compatibility no-ops (annealing is disabled but helpers might query these).
    def get_weights(self):
        return dict(self._w)

    def set_weights(self, w_dice, w_cldice, w_bce, normalize=True, epoch=None):
        if normalize:
            s = max(1e-8, w_dice + w_cldice + w_bce)
            w_dice, w_cldice, w_bce = w_dice/s, w_cldice/s, w_bce/s
        self._w.update({"w_dice": float(w_dice), "w_cldice": float(w_cldice), "w_bce": float(w_bce)})


def build_loaders(meta, batch_size):
    chans = meta.get("input_channels_used")
    if not isinstance(chans, list) or len(chans) == 0:
        raise ValueError("metadata.json missing 'input_channels_used'")

    train_set = H5IMCDataset(H5_PATH, split="train", in_channels=chans, target_channel=PERIPHERIN_CHANNEL)
    val_set   = H5IMCDataset(H5_PATH, split="val",   in_channels=chans, target_channel=PERIPHERIN_CHANNEL)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return train_loader, val_loader


def continue_run(label, run_folder, device):
    trial_path = get_best_trial(run_folder)
    if trial_path is None:
        print(f"[skip] No valid trial under: {run_folder}")
        return None

    meta_path = Path(trial_path) / "metadata.json"
    meta = json.loads(Path(meta_path).read_text())

    # Instantiate model from your helper (weights loaded, on CPU)
    model = instantiate_model(trial_path)
    if model is None:
        print(f"[skip] Could not instantiate model for: {trial_path}")
        return None

    # Resolve hparams
    lr           = float(meta.get("lr", DEFAULTS["lr"]))
    betas        = meta.get("betas", DEFAULTS["betas"])
    if isinstance(betas, str):
        try:
            betas = eval(betas)
        except Exception:
            betas = DEFAULTS["betas"]
    betas        = (float(betas[0]), float(betas[1]))
    weight_decay = float(meta.get("weight_decay", DEFAULTS["weight_decay"]))
    batch_size   = int(meta.get("batch_size", DEFAULTS["batch_size"]))
    epochs       = int(DEFAULTS["epochs"])

    # Data
    train_loader, val_loader = build_loaders(meta, batch_size)
    val_loss = validate(model, SoftDiceLoss(), val_loader, device)
    print(f"val loss before: {val_loss}")

    # Where to save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(trial_path) / f"dice_cont_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / "best_model_dice_only.pth")

    # Train (dice-only, no anneal)
    train_losses, val_losses, comp_losses, lrs = train_model(
        model=model,
        loss_function=DiceOnlyLossAdapter(),
        train_loader=train_loader,
        val_performance=SoftDiceLoss(),   # SoftDiceLoss scalar for validation
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        early_stopping=True,
        stopping_patience=STOPPING_PATIENCE,
        checkpoint_path=ckpt_path,
        weight_decay=weight_decay,
        betas=betas,
        scheduler_patience=SCHEDULER_PATIENCE,
        lr_reduction_factor=LR_REDUCTION_FACTOR,
        lr_cooldown=LR_COOLDOWN,
        min_lr=MIN_LR,
        anneal_patience=ANNEAL_PATIENCE,      # disable anneal
        unfreeze_patience=UNFREEZE_PATIENCE,  # no encoder freeze/unfreeze
        grad_clip=1.0,
    )
    print(f"val loss after: {val_losses[0]}")
    raise Exception()

    # Reload best and score thresholded hard dice
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    best_thr = find_best_threshold(model, val_loader, device)
    hard_dice = evaluate_hard_dice(model, val_loader, device, best_thr)

    # Minimal metadata
    out_meta = {
        "note": "Dice-only continuation via existing training loop (anneal disabled)",
        "source_run": run_folder,
        "best_trial": trial_path,
        "epochs_run": len(train_losses),
        "best_val_soft_dice_loss": min(val_losses) if val_losses else None,
        "best_threshold": float(best_thr),
        "val_hard_dice_at_best_threshold": float(hard_dice),
        "hparams": {
            "lr": lr, "betas": list(betas), "weight_decay": weight_decay, "batch_size": batch_size,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "stopping_patience": STOPPING_PATIENCE,
            "anneal_patience": ANNEAL_PATIENCE,
        },
        "training_curves": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lrs": lrs,
            "comp_losses": comp_losses,
        },
    }
    (out_dir / "metadata_dice_continuation.json").write_text(json.dumps(out_meta, indent=2))
    (out_dir / "summary.txt").write_text(
        f"Dice-only continuation for {label}\n"
        f"trial: {trial_path}\n"
        f"best_val_soft_dice_loss: {out_meta['best_val_soft_dice_loss']}\n"
        f"best_threshold: {best_thr:.4f}\n"
        f"val_hard_dice_at_best_threshold: {hard_dice:.4f}\n"
    )
    print(f"[done] {label}: best thr {best_thr:.3f}, val hard dice {hard_dice:.4f}")
    return str(out_dir)


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = get_device()
    print(f"Using device: {device}")

    results = {}
    for label, folder in RUNS.items():
        try:
            res = continue_run(label, folder, device)
            results[label] = res
        except Exception as e:
            print(f"[error] {label}: {e}")

    if results:
        Path("dice_only_continuations.json").write_text(json.dumps(results, indent=2))
        print("Wrote dice_only_continuations.json")


if __name__ == "__main__":
    main()
