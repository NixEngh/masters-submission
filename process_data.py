from pick import pick
from utils.constants import TIFF_DIR, MASK_DIR, DATA_DIR
import h5py
import numpy as np
from utils.vis_utils import load_case_tiff, load_tiff
from pathlib import Path
import os
from tqdm import tqdm


def remove_old():
    data_file = Path("data.h5")
    if input("are you sure? yes/[no] \n") != "yes":
        raise Exception("Cancelled by user")
    if data_file.exists():
        data_file.unlink()


def store_tiffs_as_hdf5():
    def read_ids(fn):
        return Path(fn).read_text().splitlines()[1:]

    train_ids = read_ids("meta/train_files.txt")
    val_ids = read_ids("meta/validation_files.txt")
    test_ids = read_ids("meta/test_files.txt")

    # Keep order
    case_ids = train_ids + val_ids + test_ids

    N_train, N_val, N_test = len(train_ids), len(val_ids), len(test_ids)
    N_total = len(case_ids)

    sample = load_case_tiff(train_ids[0])
    H, W, C = sample.shape

    with h5py.File("data.h5", "w") as hf:
        grp = hf.create_group("volumes")

        data = grp.create_dataset(
            "data",
            (N_total, H, W, C),
            dtype=sample.dtype,
            chunks=(1, H, W, C),
            compression="lzf",
        )

        # Create a dataset for storing case IDs
        grp.create_dataset("case_ids", data=np.array(case_ids, dtype=h5py.string_dtype()))

        for i, case_id in enumerate(tqdm(case_ids, desc="Storing TIFF files")):
            arr = load_case_tiff(case_id)  # h, w, c
            data[i] = arr

        grp.create_dataset("train_idx", data=np.arange(0, N_train, dtype="int64"))
        grp.create_dataset(
            "val_idx", data=np.arange(N_train, N_train + N_val, dtype="int64")
        )
        grp.create_dataset(
            "test_idx", data=np.arange(N_train + N_val, N_total, dtype="int64")
        )


def store_statistics():
    """Calculate and store mean and standard deviation for each channel in the dataset."""
    stats_path = Path("meta/channel_statistics.npz")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    print("Calculating channel statistics for data...")

    with h5py.File("data.h5", "r+") as hf:
        grp = hf["volumes"]
        data = grp["data"]
        train_idx = grp["train_idx"][:]
        n_channels = data.shape[-1]

        means = np.zeros(n_channels, dtype=np.float32)
        stds = np.zeros(n_channels, dtype=np.float32)

        # First pass: calculate means
        pixel_count = 0
        for i, idx in enumerate(tqdm(train_idx, desc="Computing means")):
            sample = data[idx]
            pixel_count += sample.shape[0] * sample.shape[1]

            # Sum across spatial dimensions for each channel
            channel_sums = np.sum(sample, axis=(0, 1))
            means += channel_sums

        # Normalize by total number of pixels
        means = means / pixel_count

        # Second pass: calculate standard deviations
        for i, idx in enumerate(tqdm(train_idx, desc="Computing standard deviations")):
            sample = data[idx]

            # Sum of squared differences from mean for each channel
            for c in range(n_channels):
                stds[c] += np.sum((sample[..., c] - means[c]) ** 2)

        # Normalize and take square root
        stds = np.sqrt(stds / pixel_count)

        print(f"Channel means: {means}")
        print(f"Channel standard deviations: {stds}")

        # Save statistics
        np.savez(stats_path, means=means, stds=stds)

        # Store statistics in the HDF5 file as well
        if "statistics" in hf:
            del hf["statistics"]
        stat_grp = hf.create_group("statistics")
        stat_grp.create_dataset("means", data=means)
        stat_grp.create_dataset("stds", data=stds)

        print(f"Data statistics saved to {stats_path} and added to HDF5 file")


def delete_cell_masks():
    """Delete cell masks from the HDF5 file if they exist."""
    print("Deleting cell masks from HDF5 file...")

    with h5py.File("data.h5", "r+") as hf:
        if "cell_masks" in hf:
            del hf["cell_masks"]
            print("Cell masks deleted successfully")
        else:
            print("No cell masks found in the HDF5 file")


def store_cell_masks():
    cell_mask_dir = Path(MASK_DIR())

    with h5py.File("data.h5", "r+") as hf:

        vol_group = hf["volumes"]
        mask_files = vol_group['case_ids']

        # Decode the bytes to string
        first_mask_id = mask_files[0].decode('utf-8')
        first_mask = load_tiff(str(cell_mask_dir / f"{first_mask_id}.tiff"))
        mask_shape = first_mask.shape
        mask_dtype = first_mask.dtype

        # Remove existing group if it exists
        if "cell_masks" in hf:
            del hf["cell_masks"]

        masks_grp = hf.create_group("cell_masks")

        # Create a single dataset for all masks
        masks_dataset = masks_grp.create_dataset(
            "data",
            shape=(len(mask_files), *mask_shape),
            dtype=mask_dtype,
            chunks=(1, *mask_shape),  # Chunk by individual mask
            compression="lzf",
        )

        # Fill the dataset with mask data
        for i, mask_file in enumerate(tqdm(mask_files, desc="Storing cell masks")):
            mask_id = mask_file.decode('utf-8')
            mask_path = cell_mask_dir / f"{mask_id}.tiff"
            mask_data = load_tiff(str(mask_path))
            masks_dataset[i] = mask_data

        print(f"Stored {len(mask_files)} cell masks in a single dataset")


def create_patch_volume():
    # don't implement yet
    pass


if __name__ == "__main__":

    title = "What woutd you like to do?"
    options = {
        "delete old datafile": remove_old,
        "Store tiffs to hdf5": store_tiffs_as_hdf5,
        "store statistics": store_statistics,
        "delete cell masks": delete_cell_masks,
        "store cell masks": store_cell_masks,
        "create patch group": create_patch_volume,
    }
    option, index = pick([*options.keys(), "all"], title)
    if option == "all":
        print(f"running all {len(options)}")
        for name, func in options.items():
            print(f"Starting: {name}")
            func()
            print("Done")

    else:
        print(f"Starting: {option}")
        options[option]()
        print("Done")
