from datetime import datetime
from pathlib import Path
import time
import random

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

# --- Global Settings ---
DEVICE = get_device()
RUN_ID = datetime.now().strftime("optuna_run_%Y-%m-%d_%H-%M-%S")
EPOCHS = 100
STOPPING_PATIENCE = 20
PREDICT_CHANNEL = 34
NUMBER_OF_TRIALS = 20

# --- Transfer Learning Settings ---
PERIPHERIN_CHANNEL = 31
ALL_CHANNELS = list(range(38))  
TRANSFER_INPUT_CHANNELS = [ch for ch in ALL_CHANNELS if ch not in [PERIPHERIN_CHANNEL, PREDICT_CHANNEL]]


def objective(trial: optuna.trial.Trial) -> float:
    """Defines a single trial for Optuna using plateau annealing mode."""

    # Loss weight configuration
    w_dice = trial.suggest_float("w_dice", 0.4, 0.8)
    w_cl = trial.suggest_float("w_cldice", 0.1, 0.4)
    w_bce = max(1.0 - w_dice - w_cl, 0.05)

    # Core hyperparameters
    lr = trial.suggest_float("lr", 3e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16])
    betas = eval(trial.suggest_categorical("betas", ["(0.9, 0.999)", "(0.95, 0.999)"]))
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    depth = trial.suggest_categorical("depth", [4, 5])
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.3)

    # Plateau-specific parameters
    scheduler_patience = trial.suggest_int("scheduler_patience", 6, 12)
    anneal_dice_step = trial.suggest_float("anneal_dice_step", 0.05, 0.15)

    config = {
        # Training parameters
        "lr": lr,
        "batch_size": batch_size,
        "betas": betas,
        "weight_decay": weight_decay,
        "depth": depth,
        "dropout_p": dropout_p,
        "grad_clip": 1.0,
        "bilinear": True,
        
        # Loss weights (initial)
        "w_dice": w_dice,
        "w_cldice": w_cl,
        "w_bce": w_bce,
        
        # Plateau annealing configuration
        "anneal_dice_step": anneal_dice_step,
        "anneal_max_w_dice": 0.85,
        "anneal_min_w_bce": 0.10,
        "anneal_min_w_cldice": 0.10,
        
        # Learning rate scheduling
        "scheduler_patience": scheduler_patience,
        "anneal_patience": scheduler_patience + 3,
        "lr_rewarm_factor": 1.5,
        "lr_rewarm_max": lr,
        "lr_reduction_factor": 0.7,
        "lr_cooldown": 2,
        "min_lr": max(1e-6, lr * 1e-2),
        "min_epochs_between_anneals": 3,
        
        # Transfer learning settings
        "excluded_channels": [PERIPHERIN_CHANNEL],
        "transfer_ready": True,
        "input_channels_used": TRANSFER_INPUT_CHANNELS,
    }

    trial_num = trial.number
    store_folder = Path("run") / RUN_ID / f"trial_{trial_num}"
    store_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting Trial {trial_num} | Hyperparameters: {config} ---")
    print(f"Excluding peripherin (channel {PERIPHERIN_CHANNEL}) for clean transfer learning")
    print(f"Using {len(TRANSFER_INPUT_CHANNELS)} input channels")

    # Model expects 36 input channels (37 - 1 excluded peripherin)
    model_config = {**config, "in_channels": len(TRANSFER_INPUT_CHANNELS), "out_channels": 1}
    model = UNet(**model_config)

    augmentation_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )
    
    training_set = H5IMCDataset(
        h5_path="data.h5",
        split="train",
        in_channels=TRANSFER_INPUT_CHANNELS, 
        target_channel=PREDICT_CHANNEL,
        joint_transform=augmentation_transforms,
        exclude_target_from_input=False,  
    )
    train_loader = DataLoader(
        training_set, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )

    val_set = H5IMCDataset(
        h5_path="data.h5", 
        split="val", 
        in_channels=TRANSFER_INPUT_CHANNELS,  
        target_channel=PREDICT_CHANNEL,
        exclude_target_from_input=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], num_workers=4, shuffle=False
    )

    loss_fn = WeightedDiceClDiceBCELoss(w_dice=w_dice, w_bce=w_bce, w_cldice=w_cl)

    val_performance = SoftDiceLoss()

    t0 = time.time()
    training_loss, val_loss, comp_loss, lrs = train_model(
        model,
        loss_function=loss_fn,
        train_loader=train_loader,
        val_performance=val_performance,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        early_stopping=True,
        stopping_patience=STOPPING_PATIENCE,
        checkpoint_path=f"{store_folder}/best_model.pth",
        print_iter=False,
        **config,
    )

    t1 = time.time()

    best_val_loss = min(val_loss) if val_loss else float("inf")
    print(
        f"Trial {trial_num} finished. Best soft validation loss: {best_val_loss:.4f}. Duration: {t1-t0:.2f}s"
    )

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
        "trial_number": trial_num,
        "best_threshold": best_thr,
        "val_dice_at_best_thr": val_dice,
        "epochs_run": len(training_loss),
        "training_loss": training_loss,
        "comp_loss": comp_loss,
        "val_loss": val_loss,
        "lrs": lrs,
        "train_duration": t1 - t0,
        "target_ind": PREDICT_CHANNEL,
        "loss_weight_history": getattr(loss_fn, "weight_history", []),
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
    print(f"Best trial: Trial #{study.best_trial.number} with loss {study.best_trial.value:.4f}")
    print(f"Params: {study.best_trial.params}")
