# feature_importance_perm_th_continued.py
"""
Permutation feature importance on the VALIDATION set for the chosen primary model.

Model: th_continued (continued-on-dice checkpoint)
CKPT : 
META : 

What it does
------------
- Builds the UNet exactly from trial metadata.
- Loads the dice-only continued checkpoint directly (no instantiate_model).
- Builds H5IMCDataset('val') using metadata['input_channels_used'] (same fallback as your eval).
- Baseline = mean soft_dice_coefficient(logits, target) (↑ better; no thresholding).
- For each input channel, permutes within-batch, recomputes the metric.
  Importance = baseline - permuted_mean (↑ means more important).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

# --- Hard-coded inputs (your chosen model) ---
CKPT_PATH  = Path("")
META_PATH  = Path("")
H5_PATH    = "data.h5"
PERIPHERIN_CHANNEL = 31

# Runtime controls
REPEATS      = 3          # repeats per channel permutation
MAX_BATCHES  = None       # e.g., 200 to cap runtime; None = full val
TOPK_PRINT   = 20
NUM_WORKERS  = 8
BATCH_FALLBK = 8

# --- Repo imports ---
from models.unet import UNet
from utils.data import H5IMCDataset
from utils.train_utils import get_device
from loss_functions import soft_dice_coefficient  # your existing metric (logits -> dice)

# --- Build model from metadata + load continued checkpoint ---
def load_model_from_metadata(meta_path: Path, ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"continued checkpoint not found: {ckpt_path}")

    meta = json.loads(meta_path.read_text())

    # Infer in_channels from recorded inputs; fallback like your eval
    input_channels = meta.get("input_channels_used")
    if isinstance(input_channels, list) and len(input_channels) > 0:
        in_ch = len(input_channels)
    else:
        in_ch = int(meta.get("in_channels", 37))  # safe fallback

    model = UNet(
        in_channels=in_ch,
        out_channels=meta.get("out_channels", 1),
        depth=meta.get("depth", 5),
        bilinear=meta.get("bilinear", True),
        dropout_p=meta.get("dropout_p", 0.1),
    )

    # Load the dice-only continued weights
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"Note: non-strict load due to: {e}")
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, meta

# --- Data loader (mirrors your evaluation path) ---
def build_val_loader(meta: dict, batch_size: int) -> DataLoader:
    chans = meta.get("input_channels_used")
    if not isinstance(chans, list) or len(chans) == 0:
        all_channels = list(range(38))
        predict_channel = meta.get("target_ind", PERIPHERIN_CHANNEL)
        chans = [ch for ch in all_channels if ch not in [PERIPHERIN_CHANNEL, predict_channel]]

    val_set = H5IMCDataset(H5_PATH, split="val", in_channels=chans, target_channel=PERIPHERIN_CHANNEL)
    return DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

# --- Metrics ---
@torch.no_grad()
def baseline_soft_dice(model, loader, device) -> float:
    model.eval().to(device)
    scores = []
    for xb, yb in tqdm(loader, desc="Baseline", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast(True):
            logits = model(xb)
        s = soft_dice_coefficient(logits, yb).detach().cpu()
        scores.append(s.view(-1))  # robust to scalar or per-sample returns
    return torch.cat(scores).mean().item() if scores else float("nan")

@torch.no_grad()
def permuted_soft_dice(model, loader, channel_idx: int, repeats: int, device, max_batches=None) -> float:
    model.eval().to(device)
    rep_means = []
    for r in range(repeats):
        scores = []
        seen = 0
        for xb, yb in tqdm(loader, desc=f"Permute ch{channel_idx} rep{r+1}/{repeats}", leave=False):
            xb = xb.to(device, non_blocking=True).clone()
            yb = yb.to(device, non_blocking=True)
            B = xb.size(0)
            perm = torch.randperm(B, device=xb.device)
            xb[:, channel_idx] = xb[perm, channel_idx]  # in-batch permutation
            with autocast(True):
                logits = model(xb)
            s = soft_dice_coefficient(logits, yb).detach().cpu()
            scores.append(s.view(-1))
            seen += 1
            if max_batches is not None and seen >= max_batches:
                break
        rep_means.append(torch.cat(scores).mean().item() if scores else float("nan"))
    return float(np.mean(rep_means))

# --- Main ---
def main():
    device = get_device()
    torch.backends.cudnn.benchmark = True

    model, meta = load_model_from_metadata(META_PATH, CKPT_PATH)

    bs = int(meta.get("batch_size", BATCH_FALLBK))
    val_loader = build_val_loader(meta, bs)

    input_chans = meta.get("input_channels_used")
    if not isinstance(input_chans, list) or len(input_chans) == 0:
        all_channels = list(range(38))
        predict_channel = meta.get("target_ind", PERIPHERIN_CHANNEL)
        input_chans = [ch for ch in all_channels if ch not in [PERIPHERIN_CHANNEL, predict_channel]]

    base = baseline_soft_dice(model, val_loader, device)
    print(f"\nBaseline Soft-Dice (val): {base:.4f} | inputs={len(input_chans)} chans")
    rows = []
    with torch.no_grad():
        for c in range(len(input_chans)):
            perm_mean = permuted_soft_dice(model, val_loader, c, REPEATS, device, MAX_BATCHES)
            drop = base - perm_mean
            rows.append({
                "input_idx": c,
                "orig_channel": input_chans[c],
                "perm_mean_soft_dice": perm_mean,
                "importance_drop": drop
            })
            print(f"[input {c:02d} -> orig {input_chans[c]:02d}] perm_mean={perm_mean:.4f}  drop={drop:.4f}")

    imp_df = pd.DataFrame(rows).sort_values("importance_drop", ascending=False).reset_index(drop=True)

    K = min(TOPK_PRINT, len(imp_df))
    print("\nTop channels by importance (drop in Soft-Dice):")
    with pd.option_context('display.max_rows', K, 'display.float_format', '{:.4f}'.format):
        print(imp_df.head(K).to_string(index=False))

if __name__ == "__main__":
    main()