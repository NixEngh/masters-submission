import copy
import os
import torch
from pathlib import Path
import json
from torch.amp import autocast, GradScaler
import nvsmi
import time
from loss_functions import dice_coefficient
import torch
import gc


import git


def validate(model, performance, val_loader, device):
    model.to(device)
    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

            print(
                f"Validating: Iteration {i}/{len(val_loader)}, Loss: {val_loss:.4f}",
                end="\r",
            )

            outputs = model(images)

            loss = performance(outputs, labels)
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    print()

    return avg_loss


def _anneal_loss_weights(loss_function, epoch, kwargs):
    """
    Adjusts loss function weights for plateau annealing mode.
    Increases dice weight and redistributes from BCE/CLDice proportionally.
    
    Expects 'loss_function' to implement set_weights/get_weights.
    """
    # Get current weights
    w_cur = loss_function.get_weights()
    wd, wc, wb = w_cur["w_dice"], w_cur["w_cldice"], w_cur["w_bce"]
    
    # Plateau annealing parameters
    step = float(kwargs.get("anneal_dice_step", 0.1))
    max_wd = float(kwargs.get("anneal_max_w_dice", 0.85))
    min_wb = float(kwargs.get("anneal_min_w_bce", 0.10))
    min_wc = float(kwargs.get("anneal_min_w_cldice", 0.10))

    # Calculate new dice weight (capped at maximum)
    wd_new = min(max_wd, wd + step)
    gain = wd_new - wd
    
    if gain <= 0:
        print(f"Dice weight already at maximum ({max_wd:.3f}), no annealing performed")
        return

    # Calculate available weight to redistribute
    avail_bce = max(0.0, wb - min_wb)
    avail_cldice = max(0.0, wc - min_wc)
    total_avail = avail_bce + avail_cldice
    
    if total_avail <= 0:
        print(f"Cannot redistribute weight - BCE/CLDice already at minimums")
        return
    
    # Redistribute proportionally
    if total_avail >= gain:
        # We have enough to redistribute
        take_from_bce = gain * (avail_bce / total_avail)
        take_from_cldice = gain * (avail_cldice / total_avail)
        wb_new = wb - take_from_bce
        wc_new = wc - take_from_cldice
    else:
        # Take all available weight
        wb_new = min_wb
        wc_new = min_wc
        wd_new = wd + total_avail  # Adjust dice weight to what we can actually achieve

    # Apply new weights
    loss_function.set_weights(
        w_dice=wd_new, w_cldice=wc_new, w_bce=wb_new, normalize=True, epoch=epoch
    )
    
    print(f"Annealed loss weights: dice={wd_new:.3f}, cldice={wc_new:.3f}, bce={wb_new:.3f}")


def train_model(
    model,
    loss_function,
    train_loader,
    val_performance,
    val_loader,
    device,
    epochs=500,
    lr=0.001,
    early_stopping=True,
    stopping_patience=20,
    checkpoint_path="best_model.pth",
    print_iter=False,
    **kwargs,
):
    """
    Train a PyTorch model with mixed precision, ReduceLROnPlateau scheduling, optional
    early stopping, gradient clipping.
    This routine:
    - Moves the model to the specified device.
    - Uses AdamW as the optimizer.
    - Uses torch.cuda.amp (autocast + GradScaler) for mixed-precision on CUDA.
    - Clips gradients by global norm before optimizer steps.
    - Schedules learning rate with ReduceLROnPlateau on validation loss.
    - Optionally performs early stopping with checkpointing on best validation loss.
    - Tracks and returns per-epoch training/validation losses, component losses, and LRs.
    Expected loss_function interface:
        loss, components = loss_function(outputs, labels)
    where:
        - loss is a scalar tensor used for backpropagation.
        - components is a dict with keys {"dice", "cldice", "bce"} whose values are
          accumulatable (tensors or floats); they are averaged per epoch.
    Expected validate interface:
    Parameters:
        model: torch.nn.Module
            The model to train.
        loss_function: Callable
            Callable taking (outputs, labels) and returning (loss, components_dict).
            components_dict must include keys "dice", "cldice", and "bce".
        train_loader: torch.utils.data.DataLoader
            Dataloader for training batches. Must yield (images, labels).
        val_performance: Any
            Validation metric/config object passed through to validate(...).
        val_loader: torch.utils.data.DataLoader
            Dataloader for validation data.
        device: torch.device or str
            Device to train on (e.g., "cuda", "cpu").
        epochs: int, default=500
            Maximum number of training epochs.
        lr: float, default=0.001
            Initial learning rate for AdamW.
        early_stopping: bool, default=True
            Whether to stop training when validation loss stops improving.
        stopping_patience: int, default=20
            Number of epochs with no validation improvement before early stopping.
        checkpoint_path: str, default="best_model.pth"
            File path to save the best model (lowest validation loss).
        print_iter: bool, default=False
            If True, prints per-iteration loss during training.
    Keyword Arguments (kwargs):
        weight_decay: float, default=0.01
            AdamW weight decay.
        betas: Tuple[float, float], default=(0.9, 0.999)
            AdamW betas.
        scheduler_patience: int, default=5
            Patience (epochs) for ReduceLROnPlateau before reducing LR.
        grad_clip: float, default=1.0
            Max global norm for gradient clipping.
        anneal_patience: int, default=scheduler_patience + 3
            Number of unimproved epochs after which annealing may trigger.
        lr_reduction_factor: float, default=0.7
            Factor by which to reduce LR (new_lr = old_lr * factor).
        lr_cooldown: int, default=2
            Minimum epochs between LR reductions.
        lr_rewarm_factor: float, default=1.5
            Multiplicative factor to re-warm LR after an anneal event.
        lr_rewarm_max: float, default=lr
            Upper cap for LR re-warm.
        min_lr: float, default=1e-7
            Minimum learning rate threshold.
        min_epochs_between_anneals: int, default=5
            Minimum epochs to wait between successive anneal events.
        Additional kwargs are passed to _maybe_anneal_loss if present.
    Returns:
        Tuple[
            List[float],                   # training_losses: mean training loss per epoch
            List[float],                   # val_losses: validation loss per epoch
            List[Dict[str, float]],        # comp_losses: per-epoch averages for {"dice","cldice","bce"}
            List[Union[float, List[float]]]# lrs: LR(s) per epoch; float if 1 param group else list
        ]
    Side Effects:
        - Moves the model to `device`.
        - Saves the best model state_dict to `checkpoint_path` when validation loss improves.
        - Prints training progress and early-stopping notice to stdout.
    Notes:
        - Mixed precision is enabled via autocast("cuda") and GradScaler("cuda"); ensure CUDA is
          available for best results. On CPU-only systems, consider adapting the amp usage.
        - This function relies on external helpers: `validate` and `_maybe_anneal_loss`.
    """

    model.to(device)
    print(f"Training on {device}")

    # Ensure loss/metric modules live on the right device (needed for BCE pos_weight buffers)
    if isinstance(loss_function, torch.nn.Module):
        loss_function.to(device)
    if isinstance(val_performance, torch.nn.Module):
        val_performance.to(device)

    # ==================== CONFIGURATION ====================
    # Optimizer settings
    weight_decay = kwargs.get("weight_decay", 0.01)
    betas = kwargs.get("betas", (0.9, 0.999))
    grad_clip_value = kwargs.get("grad_clip", 1.0)
    
    # Learning rate scheduling
    scheduler_patience = kwargs.get("scheduler_patience", 5)
    lr_reduction_factor = kwargs.get("lr_reduction_factor", 0.7)
    lr_cooldown = kwargs.get("lr_cooldown", 2)
    min_lr = kwargs.get("min_lr", 1e-7)
    
    # Annealing settings
    anneal_patience = kwargs.get("anneal_patience", scheduler_patience + 3)
    lr_rewarm_factor = kwargs.get("lr_rewarm_factor", 1.5)
    lr_rewarm_max = kwargs.get("lr_rewarm_max", lr)
    min_epochs_between_anneals = kwargs.get("min_epochs_between_anneals", 5)
    
    # Transfer learning: unfreezing configuration
    unfreeze_patience = kwargs.get("unfreeze_patience", 0)
    is_frozen = unfreeze_patience > 0
    # ========================================================

    # --- Optimizer and Scaler Setup ---
    if is_frozen:
        print(f"Transfer learning mode: Encoder is frozen. Unfreeze patience: {unfreeze_patience} epochs.")
        # Freeze encoder layers (inc and downs)
        for name, param in model.named_parameters():
            if 'inc' in name or 'downs' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params_to_optimize = model.parameters()

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    scaler = GradScaler("cuda")

    # State tracking
    best_val_loss = float("inf")
    epochs_since_improve = 0          # for early stopping only
    plateau_since = 0                 # for plateau-driven actions (unfreeze/anneal/LR)
    epochs_since_lr_reduction = 0
    last_anneal_epoch = -1000
    
    # Results storage
    training_losses = []
    val_losses = []
    comp_losses = []
    lrs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        epoch_comp = {
            "dice": 0,
            "cldice": 0,
            "bce": 0,
        }

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(images)
                loss, comp = loss_function(outputs, labels)

            # NEW: Add a guard to skip batches that produce non-finite loss
            if not torch.isfinite(loss):
                print(f"\n[Warning] Epoch {epoch}, Iter {i}: Non-finite loss detected ({loss.item()}). Skipping batch.")

                del loss, outputs, comp
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                continue

            # Detach components for logging *after* the check
            epoch_comp["dice"] += float(comp["dice"].detach())
            epoch_comp["cldice"] += float(comp["cldice"].detach())
            epoch_comp["bce"] += float(comp["bce"].detach())

            scaler.scale(loss).backward()

            # Apply gradient clipping before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if print_iter:
                print(
                    f"Epoch {epoch}, Iteration {i}, Loss: {loss.item():.4f}", end="\r"
                )

        print()

        for k in ("dice", "cldice", "bce"):
            v = epoch_comp[k]
            v = v / len(train_loader)
            if torch.is_tensor(v):
                v = v.item()
            else:
                v = float(v)
            epoch_comp[k] = v
        comp_losses.append(epoch_comp)

        # Log average training loss per epoch
        training_losses.append(epoch_loss / len(train_loader))

        val_loss = validate(model, val_performance, val_loader, device)
        val_losses.append(val_loss)

        # Record current LR
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        lrs.append(current_lrs[0] if len(current_lrs) == 1 else current_lrs)

        # Update improvement/plateau counters and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improve = 0
            plateau_since = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best validation loss: {val_loss:.4f}")
        else:
            epochs_since_improve += 1
            plateau_since += 1

        # --- UNFREEZE LOGIC ---
        if is_frozen and plateau_since >= unfreeze_patience:
            print(f"\n--- Epoch {epoch}: Plateau detected. Unfreezing encoder layers. ---")
            is_frozen = False
            for param in model.parameters():
                param.requires_grad = True
            print("Re-initializing optimizer for the full model.")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
            )
            plateau_since = 0
            epochs_since_lr_reduction = 0
            last_anneal_epoch = epoch - min_epochs_between_anneals

        # Update epochs since LR reduction
        epochs_since_lr_reduction += 1

        lr_reduction_due = (plateau_since >= scheduler_patience and
                            epochs_since_lr_reduction >= lr_cooldown)

        annealing_due = (plateau_since >= anneal_patience and
                         epoch - last_anneal_epoch >= min_epochs_between_anneals)

        if annealing_due:
            print(f"Triggering annealing at epoch {epoch} (plateau {plateau_since} epochs)")
            _anneal_loss_weights(loss_function, epoch, kwargs)

            # Rewarm LR
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = min(lr_rewarm_max, old_lr * lr_rewarm_factor)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"LR rewarmed from {old_lr:.6f} to {new_lr:.6f}")

            # Reset plateau only
            plateau_since = 0
            epochs_since_lr_reduction = 0
            last_anneal_epoch = epoch

        elif lr_reduction_due:
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = old_lr * lr_reduction_factor
            if new_lr >= min_lr:
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                epochs_since_lr_reduction = 0
                print(f"LR reduced from {old_lr:.6f} to {new_lr:.6f}")
            else:
                print(f"LR at minimum ({min_lr:.6f}), not reducing further")

        # Early stopping check uses only epochs_since_improve
        if early_stopping and epochs_since_improve >= stopping_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {epochs_since_improve} epochs)")
            break

        # Re-query LR after any optimizer reinit so the print is accurate
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: Train {training_losses[-1]:.4f}, Val {val_loss:.4f}, "
            f"since_best={epochs_since_improve}, plateau={plateau_since}, LR={current_lr:.6f}"
        )
        print()

    return training_losses, val_losses, comp_losses, lrs


def store_metadata(store_folder, metadata):
    meta_path: Path = store_folder / "metadata.json"

    def _to_jsonable(obj):
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return obj.item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    meta_path.write_text(json.dumps(_to_jsonable(metadata), indent=4))


def get_device():
    print("gpu id, space")
    for gpu in nvsmi.get_gpus():
        print(f"{gpu.id} | {gpu.mem_used} / {gpu.mem_total}")

    if input("automatic gpu? y/n: \n") == "y":
        while True:
            gpu_tuples = []
            for gpu in nvsmi.get_gpus():
                if gpu.mem_used < 100:
                    gpu_index = str(gpu.id)
                    break
                gpu_tuples.append((gpu.id, gpu.mem_used))
            else:
                print(f"waiting: {gpu_tuples} \r")
                time.sleep(3)
                continue
            break

    else:
        while True:
            try:
                gpu_index = int(input("Which gpu do you want to use?\n"))
                if not (0 <= gpu_index < 8):
                    raise IndexError
                break
            except ValueError:
                print("must be a number")
            except IndexError:
                print("must be a gpu index")

    return torch.device(
        f"cuda:{gpu_index if gpu_index else '0'}"
        if torch.cuda.is_available()
        else "cpu"
    )


def save_description(run_id):
    desc = input("Description:\n")
    filepath = Path("run") / run_id / "description.txt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    repo = git.Repo(search_parent_directories=True)

    filepath.write_text(desc + f"\n\n\n  Git hash: {repo.head.object.hexsha}")


def get_statistics():

    with open(os.path.join("./meta/", "statistics.json"), "r") as f:
        stats = json.load(f)

    return stats


def get_model_configs(model_class, param_grid):
    """
    Generate model configurations with hyperparameters.
    """
    configs = [{"model_name": model_class.__name__, "hyperparam_string": ""}]
    for param, values in param_grid.items():
        temp = []
        for config in configs:
            for value in values:
                newconfig = copy.deepcopy(config)
                newconfig[param] = value

                if isinstance(value, float):
                    newconfig["hyperparam_string"] += f"{param[:3]}_{value:.1e}"
                elif isinstance(value, int):
                    newconfig["hyperparam_string"] += f"{param[:3]}_{value}"
                else:
                    newconfig["hyperparam_string"] += f"{param[:3]}_{str(value)[:5]}"

                temp.append(newconfig)

        configs = temp
    for config in configs:
        config["model_class"] = model_class
    return configs


@torch.no_grad()
def find_best_threshold(model, val_loader, device, grid=None):
    if grid is None:
        grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sums = {t: 0.0 for t in grid}
    n = 0
    model.to(device).eval()
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        for t in grid:
            sums[t] += float(dice_coefficient(logits, y, threshold=t).item())
        n += 1
    return max(grid, key=lambda t: sums[t] / max(1, n))


@torch.no_grad()
def evaluate_hard_dice(model, loader, device, threshold: float):
    model.to(device).eval()
    acc, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        acc += float(dice_coefficient(model(x), y, threshold=threshold).item())
        n += 1
    return acc / max(1, n)


