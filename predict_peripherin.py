from datetime import datetime
from pathlib import Path
import time
import random
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import optuna

from loss_functions import (
    WeightedDiceClDiceBCELoss,
    SoftDiceLoss,
)
from models.unet import UNet
from utils.data import H5IMCDataset
from utils.train_utils import (
    save_description,
    store_metadata,
    get_device,
    train_model,
    find_best_threshold,
    evaluate_hard_dice,
)

# --- Experiment Configuration ---
# Set the mode for the experiment. Options: "scratch", "transfer_nf", "transfer_th"
EXPERIMENT_MODE = "transfer_th"  # "scratch", "transfer_nf", or "transfer_th"

# --- Global Settings ---
DEVICE = get_device()
RUN_ID = datetime.now().strftime(f"peripherin_{EXPERIMENT_MODE}_%Y-%m-%d_%H-%M-%S")
EPOCHS = 100
STOPPING_PATIENCE = 20
NUMBER_OF_TRIALS = 20 # Number of Optuna trials to run

# --- Channel Definitions ---
PERIPHERIN_CHANNEL = 31
NEUROFILAMENT_CHANNEL = 34 
TH_CHANNEL = 14
ALL_CHANNELS = list(range(38))

# --- Transfer Learning Configuration ---
# Set these paths to your best trained models for transfer learning.
# These should point to the 'best_model.pth' file.
NF_MODEL_PATH = "./run/optuna_run_2025-08-16_18-15-45/trial_16/best_model.pth"  # UPDATE THIS PATH
TH_MODEL_PATH = "./run/optuna_run_2025-08-16_18-12-36/trial_14/best_model.pth"  # UPDATE THIS PATH

# --- Determine settings based on experiment mode ---
PREDICT_CHANNEL = PERIPHERIN_CHANNEL
EXCLUDED_CHANNELS = []
TRANSFER_MODEL_PATH = None
INPUT_CHANNELS = ALL_CHANNELS.copy()

if EXPERIMENT_MODE == "scratch":
    # When training from scratch, we predict peripherin and exclude it from input
    EXCLUDED_CHANNELS = [PERIPHERIN_CHANNEL]
    print("Running experiment: Train from scratch to predict Peripherin.")
elif EXPERIMENT_MODE == "transfer_nf":
    # When transferring from NF, we predict peripherin.
    # The original NF model was trained without NF or Peripherin in the input.
    EXCLUDED_CHANNELS = [PERIPHERIN_CHANNEL, NEUROFILAMENT_CHANNEL]
    TRANSFER_MODEL_PATH = NF_MODEL_PATH
    print(f"Running experiment: Transfer learning from Neurofilament model ({TRANSFER_MODEL_PATH}).")
elif EXPERIMENT_MODE == "transfer_th":
    # When transferring from TH, we predict peripherin.
    # The original TH model was trained without TH or Peripherin in the input.
    EXCLUDED_CHANNELS = [PERIPHERIN_CHANNEL, TH_CHANNEL]
    TRANSFER_MODEL_PATH = TH_MODEL_PATH
    print(f"Running experiment: Transfer learning from Tyrosine Hydroxylase model ({TRANSFER_MODEL_PATH}).")

# Final input channels for the model
INPUT_CHANNELS = [ch for ch in ALL_CHANNELS if ch not in EXCLUDED_CHANNELS]
NUM_INPUT_CHANNELS = len(INPUT_CHANNELS)


def _load_transfer_arch(trial_model_path: str):
    """Read depth/bilinear/dropout from the pretrained trial's metadata.json."""
    trial_folder = Path(trial_model_path).parent
    meta_fp = trial_folder / "metadata.json"
    arch = {}
    if meta_fp.exists():
        with open(meta_fp, "r") as f:
            m = json.load(f)
        arch["depth"] = m.get("depth")
        arch["bilinear"] = m.get("bilinear")
        arch["dropout_p"] = m.get("dropout_p")
        arch["pre_in_channels"] = len(m.get("input_channels_used", [])) or None
    else:
        print(f"WARNING: metadata.json not found at {meta_fp}. Using current trial hyperparameters.")
    return arch

def _filter_state_dict_for_model(state_dict, model):
    """Keep only keys whose shapes match the current model; drop others (e.g., first conv if in_channels differ)."""
    tgt = model.state_dict()
    filtered = {}
    dropped = []
    for k, v in state_dict.items():
        if k in tgt and hasattr(v, "shape") and v.shape == tgt[k].shape:
            filtered[k] = v
        else:
            dropped.append(k)
    return filtered, dropped

def objective(trial: optuna.trial.Trial) -> float:
    """Defines a single trial for Optuna."""
    # Loss weight configuration
    w_dice = 1.0
    w_cl = 0.0
    w_bce = 0.0

    # Add pos_weight for BCE to handle class imbalance
    pos_weight = trial.suggest_float("pos_weight", 1.0, 50.0, log=True)

    # Core hyperparameters
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16])
    betas = eval(trial.suggest_categorical("betas", ["(0.9, 0.999)", "(0.95, 0.999)"]))
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    depth = trial.suggest_categorical("depth", [4, 5])
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.3)

    # Add a hyperparameter for unfreezing the encoder on plateau
    unfreeze_patience = 0
    if TRANSFER_MODEL_PATH:
        # Patience for unfreezing the encoder, relative to scheduler patience
        unfreeze_patience = trial.suggest_int("unfreeze_patience", 3, 8)


    # Plateau-specific parameters
    scheduler_patience = trial.suggest_int("scheduler_patience", 6, 12)
    anneal_dice_step = trial.suggest_float("anneal_dice_step", 0.05, 0.15)

    config = {
        "lr": lr, "batch_size": batch_size, "betas": betas, "weight_decay": weight_decay,
        "depth": depth, "dropout_p": dropout_p, "grad_clip": 1.0, "bilinear": True,
        "w_dice": w_dice, "w_cldice": w_cl, "w_bce": w_bce,
        "pos_weight": pos_weight,  # Add to config
        "anneal_dice_step": anneal_dice_step, "anneal_max_w_dice": 1,
        "anneal_min_w_bce": 0.0, "anneal_min_w_cldice": 0.0,
        "scheduler_patience": scheduler_patience, "anneal_patience": 100,
        "lr_rewarm_factor": 1.5, "lr_rewarm_max": lr, "lr_reduction_factor": 0.7,
        "lr_cooldown": 2, "min_lr": max(1e-6, lr * 1e-2), "min_epochs_between_anneals": 3,
        "excluded_channels": EXCLUDED_CHANNELS, "input_channels_used": INPUT_CHANNELS,
        "unfreeze_patience": unfreeze_patience,
    }

    trial_num = trial.number
    store_folder = Path("run") / RUN_ID / f"trial_{trial_num}"
    store_folder.mkdir(parents=True, exist_ok=True)

    # If transferring, align architecture with the pretrained metadata
    arch_overrides = {}
    if TRANSFER_MODEL_PATH:
        arch = _load_transfer_arch(TRANSFER_MODEL_PATH)
        if arch.get("depth") is not None:
            config["depth"] = arch["depth"]
        if arch.get("bilinear") is not None:
            config["bilinear"] = arch["bilinear"]
        if arch.get("dropout_p") is not None:
            config["dropout_p"] = arch["dropout_p"]

    print(f"\n--- Starting Trial {trial_num} | Hyperparameters: {config} ---")
    print(f"Predicting channel {PREDICT_CHANNEL}, using {NUM_INPUT_CHANNELS} input channels.")

    model_config = {
        "in_channels": NUM_INPUT_CHANNELS,
        "out_channels": 1,
        "depth": config["depth"],
        "bilinear": config["bilinear"],
        "dropout_p": config["dropout_p"],
    }
    model = UNet(**model_config)

    if TRANSFER_MODEL_PATH:
        try:
            print(f"Loading pre-trained weights from {TRANSFER_MODEL_PATH}")
            raw_state = torch.load(TRANSFER_MODEL_PATH, map_location="cpu", weights_only=True)
            # Filter to avoid size mismatches (e.g., different in_channels)
            filtered_state, dropped = _filter_state_dict_for_model(raw_state, model)
            if dropped:
                print(f"Transfer: dropped {len(dropped)} keys due to shape mismatch (e.g., first conv).")
            missing_cnt = len([k for k in model.state_dict().keys() if k not in filtered_state])
            print(f"Transfer: loading {len(filtered_state)} matching keys; {missing_cnt} missing will be randomly initialized.")
            model.load_state_dict(filtered_state, strict=False)
        except FileNotFoundError:
            print(f"ERROR: Pre-trained model not found at {TRANSFER_MODEL_PATH}. Aborting trial.")
            return -1.0
        except Exception as e:
            print(f"ERROR: Failed to load pre-trained weights: {e}. Aborting trial.")
            return -1.0

    augmentation_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )
    
    training_set = H5IMCDataset(
        h5_path="data.h5", split="train", in_channels=INPUT_CHANNELS, 
        target_channel=PREDICT_CHANNEL, joint_transform=augmentation_transforms,
    )
    train_loader = DataLoader(
        training_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_set = H5IMCDataset(
        h5_path="data.h5", split="val", in_channels=INPUT_CHANNELS, target_channel=PREDICT_CHANNEL,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    loss_fn = WeightedDiceClDiceBCELoss(
        w_dice=w_dice, w_bce=w_bce, w_cldice=w_cl, pos_weight=config["pos_weight"]
    )
    val_performance = SoftDiceLoss()

    t0 = time.time()
    training_loss, val_loss, comp_loss, lrs = train_model(
        model, loss_function=loss_fn, train_loader=train_loader,
        val_performance=val_performance, val_loader=val_loader, device=DEVICE,
        epochs=EPOCHS, early_stopping=True, stopping_patience=STOPPING_PATIENCE,
        checkpoint_path=f"{store_folder}/best_model.pth", print_iter=False, **config,
    )
    t1 = time.time()

    best_val_loss = min(val_loss) if val_loss else float("inf")
    print(f"Trial {trial_num} finished. Best soft validation loss: {best_val_loss:.4f}. Duration: {t1-t0:.2f}s")

    if not val_loss:
        print("No validation loss recorded, skipping evaluation.")
        return -1.0

    model.load_state_dict(
        torch.load(f"{store_folder}/best_model.pth", map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()

    best_thr = find_best_threshold(model, val_loader, DEVICE)
    val_dice = evaluate_hard_dice(model, val_loader, DEVICE, best_thr)

    print(f"Validation evaluation for Trial {trial_num}:")
    print(f"  Best threshold: {best_thr:.4f}")
    print(f"  Hard Dice at best threshold: {val_dice:.4f}\n")

    metadata = {
        "trial_number": trial_num, "best_threshold": best_thr, "val_dice_at_best_thr": val_dice,
        "epochs_run": len(training_loss), "training_loss": training_loss, "comp_loss": comp_loss,
        "val_loss": val_loss, "lrs": lrs, "train_duration": t1 - t0,
        "target_ind": PREDICT_CHANNEL, "loss_weight_history": getattr(loss_fn, "weight_history", []),
        "experiment_mode": EXPERIMENT_MODE, "transfer_model_path": TRANSFER_MODEL_PATH,
        **config,
    }
    store_metadata(store_folder, metadata)

    return val_dice


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    save_description(RUN_ID)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=NUMBER_OF_TRIALS)

    print("\n--- Optimization Finished ---")
    print(f"Best trial: Trial #{study.best_trial.number} with value {study.best_trial.value:.4f}")
    print(f"Params: {study.best_trial.params}")
