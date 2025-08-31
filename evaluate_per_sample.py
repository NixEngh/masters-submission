import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.read_runs import get_best_trial, instantiate_model
from utils.train_utils import get_device, find_best_threshold
from utils.data import H5IMCDataset
from loss_functions import ClDiceLoss, dice_coefficient, soft_dice_coefficient
from cldice import soft_cldice_loss  # added

# ---- User config ----
H5_PATH = "data.h5"
PERIPHERIN_CHANNEL = 31
BATCH_SIZE_FALLBACK = 8  # used only if batch_size missing in metadata.json

# Define the models (label, run_folder, variant)
# variant in {"annealed_best", "continued_dice_only", "standalone_dice_best"}
MODELS = [
    ("scratch_annealed", "run/peripherin_scratch_2025-08-16_20-26-08", "annealed_best"),
    ("scratch_continued", "run/peripherin_scratch_2025-08-16_20-26-08", "continued_on_dice"),
    ("nf_annealed", "run/peripherin_transfer_nf_2025-08-18_10-20-50", "annealed_best"),
    ("nf_continued", "run/peripherin_transfer_nf_2025-08-18_10-20-50", "continued_on_dice"),
    ("nf_dice_only", "run/peripherin_transfer_nf_2025-08-22_03-20-35", "standalone_dice_best"),  # NEW: NF transfer → dice-only training
    ("th_annealed", "run/peripherin_transfer_th_2025-08-19_00-55-42", "annealed_best"),
    ("th_continued", "run/peripherin_transfer_th_2025-08-19_00-55-42", "continued_on_dice"),
    ("th_dice_only", "run/peripherin_transfer_th_2025-08-22_03-21-45", "standalone_dice_best"),  # NEW: TH transfer → dice-only training
    ("scratch_dice_only", "run/peripherin_scratch_2025-08-19_08-20-01", "standalone_dice_best"),
    ("baseline", "run/baseline_runs_2025-08-16_17-58-40/BCBaseline_channel_31/default", "standalone_best"),
]

OUT_CSV = "per_sample_metrics.csv"
THRESHOLDS_JSON = "per_model_thresholds.json"


def build_val_loader(meta: dict, batch_size: int) -> DataLoader:
    chans = meta.get("input_channels_used")
    if not isinstance(chans, list) or len(chans) == 0:
        raise ValueError("metadata.json missing 'input_channels_used'")

    val_set = H5IMCDataset(H5_PATH, split="val", in_channels=chans, target_channel=PERIPHERIN_CHANNEL)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False,
                      num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)


def find_latest_cont_folder(trial_path: Path) -> Path | None:
    cands = sorted([p for p in trial_path.glob("dice_cont_*") if p.is_dir()])
    return cands[-1] if cands else None


def load_model_for_variant(trial_path: Path, variant: str):
    # Returns (model, metadata_dict, checkpoint_path_string)
    meta_path = trial_path / "metadata.json"
    meta = json.loads(Path(meta_path).read_text())
    model = instantiate_model(trial_path)  # loads best_model.pth by default (annealed best)
    ckpt = trial_path / "best_model.pth"

    if variant == "annealed_best":
        return model, meta, str(ckpt)

    if variant == "continued_on_dice":
        cont = find_latest_cont_folder(trial_path)
        if cont is None:
            raise FileNotFoundError(f"No continuation folder under {trial_path}")
        # reload model weights from the dice-only continuation checkpoint
        model.load_state_dict(torch.load(cont / "best_model_dice_only.pth", map_location="cpu"))
        return model, meta, str(cont / "best_model_dice_only.pth")

    if variant == "standalone_dice_best":
        # For standalone dice-only experiment, best_model.pth is the correct checkpoint already
        return model, meta, str(ckpt)

    if variant == "standalone_best":
        # For other standalone experiments, best_model.pth is the correct checkpoint
        # Reconstruct input channels if they are missing (case for baselines)
        if "input_channels_used" not in meta:
            print(f"Reconstructing input channels for baseline {trial_path.parent.name}")
            all_channels = list(range(38))
            predict_channel = meta.get("target_ind")
            if predict_channel is None:
                raise ValueError("Baseline metadata missing 'target_ind'")
            # Logic from store_baselines.py
            meta["input_channels_used"] = [
                ch for ch in all_channels if ch not in [PERIPHERIN_CHANNEL, predict_channel]
            ]
        return model, meta, str(ckpt)

    raise ValueError(f"Unknown variant: {variant}")


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = get_device()
    cldice_loss = ClDiceLoss()  # metric will be 1 - loss

    out_lines = ["model_label,sample_idx,soft_dice,hard_dice,cldice"]
    thr_dict = {}

    for label, run_folder, variant in MODELS:
        if variant == "standalone_best":
            trial_path = Path(run_folder)
        else:
            trial_path = get_best_trial(run_folder)
        if trial_path is None:
            print(f"[skip] No valid trial in {run_folder}")
            continue

        model, meta, ckpt_used = load_model_for_variant(Path(trial_path), variant)
        model.to(device).eval()

        # Build val loader (batch size from metadata if present)
        bs = int(meta.get("batch_size", BATCH_SIZE_FALLBACK))
        val_loader = build_val_loader(meta, bs)

        # Find threshold on the *whole* val set
        best_thr = float(find_best_threshold(model, val_loader, device))
        thr_dict[label] = {"threshold": best_thr, "trial_path": trial_path, "variant": variant, "ckpt": ckpt_used}
        print(f"[{label}] best_thr={best_thr:.3f} ckpt={ckpt_used}")

        # Per-sample metrics
        with torch.no_grad():
            sample_idx = 0
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                # H5IMCDataset yields targets as (N,1,H,W); keep as-is
                yb = yb.to(device, non_blocking=True).float()

                logits = model(xb)  # (N,1,H,W)

                for i in range(logits.size(0)):
                    l = logits[i:i+1].contiguous()  # (1,1,H,W)
                    g = yb[i:i+1].contiguous()      # (1,1,H,W)

                    # Metrics
                    soft = soft_dice_coefficient(l, g).item()
                    hard = dice_coefficient(l, g, threshold=best_thr).item()
                    cld  = (1.0 - cldice_loss(l, g)).item()  # or: 1 - soft_cldice_loss(torch.sigmoid(l), g).item()

                    out_lines.append(f"{label},{sample_idx},{soft:.6f},{hard:.6f},{cld:.6f}")
                    sample_idx += 1

    Path(OUT_CSV).write_text("\n".join(out_lines))
    Path(THRESHOLDS_JSON).write_text(json.dumps(thr_dict, indent=2, default=str))

    print(f"Wrote {OUT_CSV} and {THRESHOLDS_JSON}")


if __name__ == "__main__":
    main()
