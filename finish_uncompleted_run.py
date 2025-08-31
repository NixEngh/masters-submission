
# finish_uncompleted_run.py
# ------------------------------------------------------------
# Finish an uncompleted Optuna run by replaying completed trials
# and then continuing with only the remaining trials.
#
# USAGE: Set RUN_DIR and TOTAL_TRIALS below, then run:
#   python finish_uncompleted_run.py
#
# This file must live next to predict_peripherin.py (it imports from it).
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
import sys
import random
import optuna
from optuna.trial import create_trial, TrialState
from optuna.distributions import (
    FloatDistribution, IntDistribution, CategoricalDistribution
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
RUN_DIR = Path("")  #
TOTAL_TRIALS = 20                                                  # planned total
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Ensure we can import predict_peripherin.py from the same directory
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Import the user's training script (objective + helpers)
import predict_peripherin as pp  # noqa: E402


def _infer(key: str):
    """Return the first non-null value of `key` found in any trial metadata."""
    for meta_path in sorted(RUN_DIR.glob("trial_*/metadata.json")):
        try:
            m = json.loads(meta_path.read_text())
            if key in m and m[key] is not None:
                return m[key]
        except Exception:
            pass
    return None


def _read_completed_trial_metas():
    for meta_path in sorted(RUN_DIR.glob("trial_*/metadata.json")):
        try:
            yield meta_path, json.loads(meta_path.read_text())
        except Exception as e:
            print(f"[replay] WARNING: Failed reading {meta_path}: {e}")


def _make_exact_distributions(include_unfreeze: bool) -> dict:
    """
    Copy the EXACT hyperparameter names and ranges from predict_peripherin.py.
    This mirrors the suggest_* calls in your objective.
    """
    d = {
        "w_dice": FloatDistribution(0.4, 0.8),
        "w_cldice": FloatDistribution(0.0, 0.4),
        "pos_weight": FloatDistribution(1.0, 50.0, log=True),
        "lr": FloatDistribution(3e-4, 3e-3, log=True),
        "batch_size": CategoricalDistribution([16]),
        "betas": CategoricalDistribution(["(0.9, 0.999)", "(0.95, 0.999)"]),
        "weight_decay": FloatDistribution(1e-4, 1e-1, log=True),
        "depth": CategoricalDistribution([4, 5]),
        "dropout_p": FloatDistribution(0.0, 0.3),
        "scheduler_patience": IntDistribution(6, 12),
        "anneal_dice_step": FloatDistribution(0.05, 0.15),
    }
    if include_unfreeze:
        d["unfreeze_patience"] = IntDistribution(3, 8)
    return d


def _params_from_meta(m: dict, include_unfreeze: bool) -> dict:
    """Extract ONLY the params that your objective suggests, mapping types exactly."""
    # betas may be saved as list/tuple; convert to the string choice used in suggest_categorical
    betas_val = m.get("betas")
    if isinstance(betas_val, (list, tuple)):
        betas_str = f"({betas_val[0]}, {betas_val[1]})"
    else:
        betas_str = str(betas_val) if betas_val is not None else "(0.9, 0.999)"
    params = {
        "w_dice": m["w_dice"],
        "w_cldice": m["w_cldice"],
        "pos_weight": m["pos_weight"],
        "lr": m["lr"],
        "batch_size": m.get("batch_size", 16),
        "betas": betas_str,
        "weight_decay": m["weight_decay"],
        "depth": m["depth"],
        "dropout_p": m["dropout_p"],
        "scheduler_patience": m.get("scheduler_patience", 10),
        "anneal_dice_step": m.get("anneal_dice_step", 0.10),
    }
    if include_unfreeze and "unfreeze_patience" in m:
        params["unfreeze_patience"] = m["unfreeze_patience"]
    return params


def _metric_from_meta(m: dict) -> float:
    for k in ("val_dice_at_best_thr", "val_soft_dice", "val_metric", "best_val"):
        if k in m and m[k] is not None:
            return float(m[k])
    raise ValueError("No suitable validation metric found in metadata.")


def main():
    assert RUN_DIR.exists(), f"Run directory not found: {RUN_DIR}"

    # Make predict_peripherin write new trials into the SAME folder
    try:
        pp.RUN_ID = RUN_DIR.name
    except Exception:
        pass

    # Keep original experiment mode & transfer weights for consistency
    exp_mode = _infer("experiment_mode")
    if exp_mode:
        try:
            pp.EXPERIMENT_MODE = exp_mode
            print(f"[setup] EXPERIMENT_MODE = {pp.EXPERIMENT_MODE}")
        except Exception:
            pass
    transfer_path = _infer("transfer_model_path")
    if transfer_path:
        try:
            pp.TRANSFER_MODEL_PATH = transfer_path
            print(f"[setup] TRANSFER_MODEL_PATH = {pp.TRANSFER_MODEL_PATH}")
        except Exception:
            pass

    # Seed (objective may also seed internally)
    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    except Exception:
        pass

    # Build a study and "replay" completed trials with EXACT dists
    include_unfreeze = bool(getattr(pp, "TRANSFER_MODEL_PATH", None))
    dists = _make_exact_distributions(include_unfreeze)
    study = optuna.create_study(direction="maximize")

    completed = 0
    for meta_path, m in _read_completed_trial_metas():
        try:
            params = _params_from_meta(m, include_unfreeze)
            trial = create_trial(
                params=params,
                distributions={k: dists[k] for k in params.keys()},
                value=_metric_from_meta(m),
                state=TrialState.COMPLETE,
                user_attrs={"replayed_from": str(meta_path)},
            )
            study.add_trial(trial)
            completed += 1
        except Exception as e:
            print(f"[replay] WARNING: Skipping {meta_path} due to: {e}")

    to_run = max(TOTAL_TRIALS - completed, 0)
    print(f"[plan] Completed: {completed} | Target total: {TOTAL_TRIALS} | Will run: {to_run}")

    if to_run == 0:
        print("[done] Nothing to do — all trials already completed.")
        if len(study.trials) > 0:
            print(f"Best trial: #{study.best_trial.number} value={study.best_trial.value:.4f}")
            print(f"Params: {study.best_trial.params}")
        return

    # Optional: describe the run folder
    try:
        pp.save_description(RUN_DIR.name)
    except Exception:
        pass

    # Run only the remaining trials; catch common train-time errors
    study.optimize(pp.objective, n_trials=to_run, catch=(RuntimeError,))

    print("\n--- Optimization Finished ---")
    if len(study.trials) > 0:
        print(f"Best trial: #{study.best_trial.number} value={study.best_trial.value:.4f}")
        print(f"Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
