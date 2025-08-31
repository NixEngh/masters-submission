import os
import json
import pandas as pd
from models.unet import UNet
from models.bcbaseline import BCBaseline
import torch
from utils.data import H5IMCDataset
from utils.train_utils import get_statistics
import matplotlib.pyplot as plt




# Update your helper to include a trial_id for cleaner labels
def get_metadata_dict_optuna(folder):
    import os
    metadata_dict = {}

    for trial in os.listdir(folder):
        trial_folder = os.path.join(folder, trial)

        if os.path.isdir(trial_folder):
            metadata_file = os.path.join(trial_folder, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    metadata["trial_id"] = os.path.basename(trial_folder)
                    metadata_dict[f"{folder}/{trial_folder}"] = metadata

                    best_model_path = os.path.join(trial_folder, 'best_model.pth')
                    if os.path.exists(best_model_path):
                        metadata["model_path"] = best_model_path
    return metadata_dict

def get_optuna_df(folder):
    met_dict = get_metadata_dict_optuna(folder)
    if not met_dict:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(met_dict, orient="index")
    df.index.name = "run_id"
    df.sort_index(inplace=True)
    return df


def get_metadata_dict_peripherin(folder: str) -> dict:
    """
    Collect metadata for peripherin runs produced by predict_peripherin.py.

    Expects `folder` to contain trial subfolders (e.g., trial_0, trial_1, ...),
    each with metadata.json and optionally best_model.pth.

    Returns a dict keyed by "<run_name>/<trial_folder_name>" with loaded
    metadata and convenience fields. If best_model.pth exists, attaches a
    constructed UNet under metadata["model"] with in_channels inferred from
    metadata["input_channels_used"].
    """
    import os
    import json
    out = {}


    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    run_name = os.path.basename(os.path.normpath(folder))

    for trial_id in sorted(os.listdir(folder)):
        trial_folder = os.path.join(folder, trial_id)
        if not os.path.isdir(trial_folder):
            continue

        meta_file = os.path.join(trial_folder, "metadata.json")
        if not os.path.exists(meta_file):
            continue

        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {meta_file}: {e}")
            continue

        # Convenience fields
        meta["trial_id"] = trial_id
        meta["trial_folder"] = trial_folder
        meta["run_folder"] = folder

        # Try to instantiate and attach the trained model
        best_model_path = os.path.join(trial_folder, "best_model.pth")
        if os.path.exists(best_model_path):
            try:
                # Infer in_channels from recorded inputs; fallback to meta.get("in_channels") or 36
                input_channels = meta.get("input_channels_used")
                if isinstance(input_channels, list) and len(input_channels):
                    in_ch = len(input_channels)
                else:
                    in_ch = int(meta.get("in_channels", 36))

                # model = UNet(
                #     in_channels=in_ch,
                #     out_channels=1,
                #     depth=meta.get("depth", 4),
                #     bilinear=meta.get("bilinear", True),
                #     dropout_p=meta.get("dropout_p", 0.0),
                # )

                # state_dict = torch.load(best_model_path, weights_only=True, map_location="cpu")
                # try:
                #     model.load_state_dict(state_dict, strict=True)
                # except Exception as e:
                #     print(f"Note: non-strict load for {trial_id} due to: {e}")
                #     model.load_state_dict(state_dict, strict=False)
                # meta["model"] = model
                meta["best_model_path"] = best_model_path
            except Exception as e:
                print(f"Warning: failed to load model for {trial_id}: {e}")

        key = f"{run_name}/{trial_id}"
        out[key] = meta

    return out

def get_peripherin_df(folder: str) -> pd.DataFrame:
    """
    Build a DataFrame from get_metadata_dict_peripherin(folder).
    Index = 'run_id' in the form '<run_name>/<trial_id>'.
    """
    met_dict = get_metadata_dict_peripherin(folder)
    if not met_dict:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(met_dict, orient="index")
    df.index.name = "run_id"
    df.sort_index(inplace=True)
    return df

def get_metadata_dict_baselines(folder: str) -> dict:
    """
    Walk a baseline_runs folder (e.g., ./run/baseline_runs_YYYY-mm-dd_HH-MM-SS/) and
    collect per-trial metadata.

    Returns a dict keyed by "<model_name>/<trial_folder_name>" with the loaded
    metadata.json contents plus some convenience fields:
      - model_name: name of the model folder (e.g., BCBaseline_channel_14)
      - trial_id: the subfolder name for the trial/hyperparameters
      - run_folder: the input folder you passed in
      - trial_folder: absolute path to the trial folder
      - best_model_path: path to best_model.pth if present

    Notes
    - This function does not instantiate models; it only aggregates metadata.
    - It tolerates missing files and continues.
    """
    import os
    import json

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    out = {}
    # Iterate model folders under the baseline run folder
    for model_name in sorted(os.listdir(folder)):
        model_path = os.path.join(folder, model_name)
        if not os.path.isdir(model_path):
            continue

        # Each model folder contains multiple trials/hparam folders
        for trial_id in sorted(os.listdir(model_path)):
            trial_path = os.path.join(model_path, trial_id)
            if not os.path.isdir(trial_path):
                continue

            meta_file = os.path.join(trial_path, "metadata.json")
            if not os.path.exists(meta_file):
                # Skip folders without metadata.json
                continue

            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"Warning: failed to read {meta_file}: {e}")
                continue

            # Attach convenience fields
            meta["model_name"] = model_name
            meta["trial_id"] = trial_id
            meta["run_folder"] = folder
            meta["trial_folder"] = trial_path

            best_model_path = os.path.join(trial_path, "best_model.pth")
            if os.path.exists(best_model_path):
                meta["best_model_path"] = best_model_path

            key = f"{model_name}/{trial_id}"
            out[key] = meta

    return out

def get_baseline_df(folder: str) -> pd.DataFrame:
    """
    Build a DataFrame from get_metadata_dict_baselines(folder).
    Index = 'run_id' in the form '<model_name>/<trial_id>'.
    """
    met_dict = get_metadata_dict_baselines(folder)
    if not met_dict:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(met_dict, orient="index")
    df.index.name = "run_id"
    df.sort_index(inplace=True)
    return df


def instantiate_model(trial_path):
    """
    Instantiates a model from a trial folder.

    Reads metadata.json to configure the model architecture and then loads
    the weights from best_model.pth. Handles UNet and BCBaseline.

    Args:
        trial_path (str): Path to the trial directory containing metadata.json
                          and best_model.pth.

    Returns:
        torch.nn.Module: The instantiated model with loaded weights, on the CPU.
                         Returns None if essential files are missing.
    """
    meta_path = os.path.join(trial_path, "metadata.json")
    model_path = os.path.join(trial_path, "best_model.pth")

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        print(f"Warning: Missing metadata.json or best_model.pth in {trial_path}")
        return None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        model_class_name = meta.get("model_class", "UNet") # Default to UNet for old runs

        if model_class_name == "BCBaseline":
            model = BCBaseline(
                in_channels=meta.get("in_channels"),
                out_channels=meta.get("out_channels", 1)
            )
        else: # Assumes UNet
            # Infer model hyperparameters from metadata with sensible defaults
            input_channels = meta.get("input_channels_used")
            if isinstance(input_channels, list) and len(input_channels) > 0:
                in_ch = len(input_channels)
            else:
                # Fallback for older runs or different metadata structures
                in_ch = int(meta.get("in_channels", 37))

            model = UNet(
                in_channels=in_ch,
                out_channels=meta.get("out_channels", 1),
                depth=meta.get("depth", 5),
                bilinear=meta.get("bilinear", True),
                dropout_p=meta.get("dropout_p", 0.1),
            )

        # Load the state dictionary
        state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        
        # Load weights into the model
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Note: Non-strict state_dict load for {trial_path} due to: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()  # Set model to evaluation mode
        return model

    except Exception as e:
        print(f"Error instantiating model from {trial_path}: {e}")
        return None


def get_best_trial(run_path):
    """
    Finds the best trial in a run based on 'best_hard_dice_at_threshold'.

    Args:
        run_path (str): Path to the run directory containing trial subfolders.

    Returns:
        str: The path to the best trial directory, or None if no valid trials are found.
    """
    if not os.path.isdir(run_path):
        print(f"Error: Run path not found: {run_path}")
        return None

    best_score = -1.0
    best_trial_path = None


    for trial_id in os.listdir(run_path):
        trial_path = os.path.join(run_path, trial_id)
        if not os.path.isdir(trial_path):
            continue

        meta_path = os.path.join(trial_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            score = metadata.get("val_dice_at_best_thr")

            if score is not None and score > best_score:
                best_score = score
                best_trial_path = trial_path
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process metadata for {trial_path}: {e}")
            continue

    if best_trial_path is None:
        
        print(f"Warning: No valid trials with 'val_dice_at_best_threshold' found in {run_path}")

    return best_trial_path