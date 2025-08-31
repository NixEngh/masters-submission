import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import torchvision.transforms as transforms  # NEW

from models.bcbaseline import BCBaseline
from utils.data import H5IMCDataset
from loss_functions import WeightedDiceClDiceBCELoss, SoftDiceLoss  # CHANGE
from utils.train_utils import (
    train_model,
    get_device,
    store_metadata,
    find_best_threshold,
    evaluate_hard_dice,
)

PERIPHERIN_CHANNEL = 31  # NEW
ALL_CHANNELS = list(range(38))  # NEW

def run_single_baseline(channel, run_id, device):
    """
    Trains a single instance of the BCBaseline model for a specific channel.
    Args:
        channel (int): The index of the channel to be predicted.
        run_id (str): The main identifier for this batch of baseline runs.
    """
    # Each baseline gets its own subfolder within the main run folder
    store_folder = Path("run") / run_id / f"BCBaseline_channel_{channel}" / "default"
    store_folder.mkdir(parents=True, exist_ok=True)

    predict_channel = channel
    # Match run_optuna input selection (exclude peripherin + target)
    transfer_input_channels = [ch for ch in ALL_CHANNELS if ch not in [PERIPHERIN_CHANNEL, predict_channel]]  # NEW

    # 2. Define a reasonable, fixed configuration for the baseline
    config = {
        'model_name': f'BCBaseline_channel_{channel}',
        'lr': 1e-3,
        'weight_decay': 0.01,
        'bce_pos_weight': 1.0,
        'bce_weight': 0.5,
        'dice_weight': 0.5,
        'batch_size': 16,
        'in_channels': len(transfer_input_channels),  # CHANGE
        'out_channels': 1,
    }
    epochs = 100
    stopping_patience = 15

    # 3. Setup DataLoaders (match joint aug and channel selection)
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])  # NEW

    training_set = H5IMCDataset(
        h5_path="data.h5",
        split="train",
        in_channels=transfer_input_channels,      # NEW
        target_channel=predict_channel,
        joint_transform=augmentation_transforms,  # NEW
        exclude_target_from_input=False,          # NEW (since in_channels provided)
    )
    train_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    val_set = H5IMCDataset(
        h5_path="data.h5",
        split="val",
        in_channels=transfer_input_channels,      # NEW
        target_channel=predict_channel,
        exclude_target_from_input=False,          # NEW
    )
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 4. Instantiate Model and Loss
    model = BCBaseline(in_channels=config['in_channels'], out_channels=config['out_channels'])

    # Use tuple-returning loss to match train_model in run_optuna
    loss_function = WeightedDiceClDiceBCELoss(   # CHANGE
        w_dice=config['dice_weight'],
        w_cldice=0.0,                  # baseline: no clDice
        w_bce=config['bce_weight'],
    )
    val_performance = SoftDiceLoss()

    print(f"--- Training Baseline for Channel {channel} ---")

    # 5. Train the model
    t0 = time.time()
    training_loss, val_loss, comp_loss, lrs = train_model(
        model,
        loss_function=loss_function,
        train_loader=train_loader,
        val_performance=val_performance,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        early_stopping=True,
        stopping_patience=stopping_patience,
        checkpoint_path=f"{store_folder}/best_model.pth",
        print_iter=True,
        **config,
    )
    t1 = time.time()
    print(f"\nBaseline training for channel {channel} finished in {t1-t0:.2f} seconds.")

    # Load best checkpoint and evaluate best threshold + hard Dice (to mirror run_optuna.py)
    model.load_state_dict(
        torch.load(f"{store_folder}/best_model.pth", map_location=device, weights_only=True)
    )
    model.to(device).eval()

    best_thr = find_best_threshold(model, val_loader, device)
    val_dice = evaluate_hard_dice(model, val_loader, device, best_thr)
    print(f"Validation evaluation for channel {channel}:")
    print(f"  Best threshold: {best_thr:.4f}")
    print(f"  Hard Dice at best threshold: {val_dice:.4f}\n")

    # 6. Save metadata
    metadata = {
        "epochs_run": len(training_loss),
        "training_loss": training_loss,
        "val_loss": val_loss,
        "comp_loss": comp_loss,
        "lrs": lrs,
        "best_threshold": best_thr,
        "val_dice_at_best_thr": val_dice,
        "train_duration": t1 - t0,
        "target_ind": predict_channel,
        **config,
        "loss_weight_history": getattr(loss_function, "weight_history", []),
        "model_class": "BCBaseline",
    }
    store_metadata(store_folder, metadata)
    print(f"Results saved to: {store_folder}\n")

if __name__ == "__main__":
    # Create a single run ID for this entire batch of baseline trainings
    main_run_id = f"baseline_runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # 1. Setup paths and device
    device = get_device()
    
    channels_to_run = [14, 31, 34]
    for channel in channels_to_run:
        run_single_baseline(channel=channel, run_id=main_run_id, device=device)
    
    print("--- All baseline runs completed. ---")
