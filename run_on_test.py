# test_eval_th_continued.py
"""
Evaluate the chosen best model (continued-on-dice) on the TEST set.

CKPT: run/peripherin_transfer_th_2025-08-19_00-55-42/trial_12/dice_cont_2025-08-20_18-37-18/best_model_dice_only.pth
META: run/peripherin_transfer_th_2025-08-19_00-55-42/trial_12/metadata.json
Label: th_continued
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Hard-coded choices ---
MODEL_LABEL = "th_continued"
CKPT_PATH = Path("run/peripherin_transfer_th_2025-08-19_00-55-42/trial_12/dice_cont_2025-08-20_18-37-18/best_model_dice_only.pth")
META_PATH = Path("run/peripherin_transfer_th_2025-08-19_00-55-42/trial_12/metadata.json")
THRESHOLDS_JSON = Path("per_model_thresholds.json")
OUT_DIR = Path("eval_out/test_set"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "per_sample_th_continued_test.csv"

# --- Data constants ---
H5_PATH = "data.h5"
PERIPHERIN_CHANNEL = 31
BATCH_SIZE_FALLBACK = 8
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT = True
PREFETCH = 2

from models.unet import UNet
from utils.data import H5IMCDataset
from utils.train_utils import get_device
from loss_functions import soft_dice_per_sample, dice_coefficient  # reuse your metrics

# Optional clDice (only if you export it)
try:
    from loss_functions import cldice_coefficient as _cldice_fn
except Exception:
    _cldice_fn = None

# --- Helpers ---
def load_threshold(label: str) -> float:
    if THRESHOLDS_JSON.exists():
        try:
            data = json.loads(THRESHOLDS_JSON.read_text())
            if label in data:
                return float(data[label])
        except Exception:
            pass
    return 0.20

def build_test_loader(meta: dict, batch_size: int) -> DataLoader:
    chans = meta.get("input_channels_used")
    if not isinstance(chans, list) or not chans:
        all_channels = list(range(38))
        predict_channel = meta.get("target_ind", PERIPHERIN_CHANNEL)
        chans = [ch for ch in all_channels if ch not in [PERIPHERIN_CHANNEL, predict_channel]]
    ds = H5IMCDataset(H5_PATH, split="test", in_channels=chans, target_channel=PERIPHERIN_CHANNEL)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT, prefetch_factor=PREFETCH
    )

def build_model_from_meta(meta_path: Path, ckpt_path: Path, device) -> tuple[torch.nn.Module, dict]:
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    meta = json.loads(meta_path.read_text())

    if isinstance(meta.get("input_channels_used"), list) and meta["input_channels_used"]:
        in_ch = len(meta["input_channels_used"])
    else:
        in_ch = int(meta.get("in_channels", 37))

    model = UNet(
        in_channels=in_ch,
        out_channels=meta.get("out_channels", 1),
        depth=meta.get("depth", 5),
        bilinear=meta.get("bilinear", True),
        dropout_p=meta.get("dropout_p", 0.1),
    )

    state = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[warn] non-strict load due to: {e}")
        model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, meta

def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, 2.5)
    hi = np.percentile(boots, 97.5)
    return x.mean(), lo, hi

# --- Main ---
def main():
    device = get_device()
    torch.backends.cudnn.benchmark = True

    model, meta = build_model_from_meta(META_PATH, CKPT_PATH, device)
    bs = int(meta.get("batch_size", BATCH_SIZE_FALLBACK))
    test_loader = build_test_loader(meta, bs)

    thr = load_threshold(MODEL_LABEL)
    print(f"Using threshold for '{MODEL_LABEL}': {thr:.3f}")

    soft_dice_list, hard_dice_list, cldice_list, sample_ids = [], [], [], []

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(xb)
                # Soft-Dice on logits (your metric)
                sd = soft_dice_per_sample(logits, yb).detach().cpu().view(-1).numpy()

            # Hard Dice via your dice_coefficient(threshold=…); note it returns a batch mean,
            # so compute per-sample explicitly for the CSV:
            probs = torch.sigmoid(logits)
            pred_bin = (probs >= thr).to(yb.dtype)

            # Per-sample hard dice (replicate your formula to get per-slide values)
            inter = (pred_bin * yb).sum(dim=(-2, -1))
            denom = pred_bin.sum(dim=(-2, -1)) + yb.sum(dim=(-2, -1))
            hd = ((2 * inter + 1e-6) / (denom + 1e-6)).detach().cpu().view(-1).numpy()

            soft_dice_list.append(sd); hard_dice_list.append(hd)

            if _cldice_fn is not None:
                try:
                    cd = _cldice_fn(pred_bin, yb).detach().cpu().view(-1).numpy()
                except Exception:
                    cd = np.full(sd.shape, np.nan)
                cldice_list.append(cd)

            B = sd.shape[0]
            start = len(sample_ids)
            sample_ids.extend(range(start, start + B))

    soft_dice_arr = np.concatenate(soft_dice_list) if soft_dice_list else np.array([], dtype=float)
    hard_dice_arr = np.concatenate(hard_dice_list) if hard_dice_list else np.array([], dtype=float)
    cldice_arr = np.concatenate(cldice_list) if _cldice_fn is not None and cldice_list else None

    out = {
        "sample_idx": sample_ids[: len(soft_dice_arr)],
        "model_label": [MODEL_LABEL] * len(soft_dice_arr),
        "soft_dice": soft_dice_arr,
        "hard_dice_at_thr": hard_dice_arr,
    }
    if cldice_arr is not None:
        out["cldice"] = cldice_arr

    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-sample metrics → {OUT_CSV}")

    def _summ(name: str, arr: np.ndarray):
        m, lo, hi = _bootstrap_mean_ci(arr)
        med = float(np.median(arr)); q1 = float(np.percentile(arr, 25)); q3 = float(np.percentile(arr, 75))
        print(f"{name}: Mean {m:.3f} [95% CI {lo:.3f}, {hi:.3f}] | Median {med:.3f} [IQR {q1:.3f}, {q3:.3f}] | n={arr.size}")

    print("\n=== TEST summary (per slide) ===")
    _summ("Soft-Dice", soft_dice_arr)
    _summ("Hard Dice @ thr", hard_dice_arr)
    if cldice_arr is not None:
        _summ("clDice", cldice_arr)

if __name__ == "__main__":
    main()
