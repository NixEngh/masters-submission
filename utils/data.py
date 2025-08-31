import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from utils.vis_utils import load_tiff
from utils.constants import TIFF_DIR
from pathlib import Path
import numpy as np
import json
import h5py

filenames = {
    "train": "train_files.txt",
    "validation": "validation_files.txt",
    "test": "test_files.txt",
}


class IMCDataset(Dataset):
    def __init__(
        self,
        type="train",
        train_test_dir="./meta/",
        normalize=True,
        normalize_output=True,
        feature_inds=None,
        label_idx=31,
        cap_inputs=False,
        cap_target=False,
    ):
        if not (0 <= label_idx < 38):
            label_idx = 31

        self.tiff_dir = TIFF_DIR()
        filename = filenames[type]
        self.cases = (
            Path(os.path.join(train_test_dir, filename)).read_text().splitlines()[2:]
        )
        self.feature_inds = feature_inds
        self.label_idx = label_idx

        if normalize or normalize_output:
            with open(os.path.join(train_test_dir, "statistics.json"), "r") as f:
                stats = json.load(f)

            all_means = stats["mean"]
            all_stds = stats["std"]

            self.target_mean = all_means[self.label_idx]
            self.target_std = all_stds[self.label_idx]

            if self.feature_inds:
                self.mean = [all_means[i] for i in self.feature_inds]
                self.std = [all_stds[i] for i in self.feature_inds]
            else:
                # Otherwise use all means/stds except the label index
                self.mean = np.delete(all_means, self.label_idx).tolist()
                self.std = np.delete(all_stds, self.label_idx).tolist()

        transform_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
        ]
        if normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.transform = self.transform = transforms.Compose(transform_list)

        target_transforms = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
        ]

        if normalize_output:
            target_transforms.append(
                transforms.Normalize(mean=self.target_mean, std=self.target_std),
            )

        self.target_transform = transforms.Compose(target_transforms)

        self.cap_inputs = cap_inputs

        self.cap_target = cap_target

        self.k = 3
        self.threshold = self.k
        if not normalize_output and cap_target:
            self.threshold = self.target_mean + self.k * self.target_std

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        img_path = os.path.join(self.tiff_dir, f"{self.cases[idx]}.tiff")
        tiff = load_tiff(img_path, transpose=True)[1:-1, 1:-1, :]

        label = tiff[:, :, self.label_idx]
        if self.feature_inds:
            image = tiff[:, :, self.feature_inds]
        else:
            image = np.delete(tiff, self.label_idx, axis=2)

        image = self.transform(image)
        label = self.target_transform(label)

        if self.cap_target:
            label = torch.clamp(label, max=self.threshold)

        return image, label

    def get_case_from_index(self, idx):
        return self.cases[idx]


class IMCDataset2(Dataset):
    def __init__(
        self,
        type="train",
        hd_file="data.h5",
        normalize=True,
        normalize_output=True,
        feature_inds=None,
        label_idx=31,
        use_capped=True,
    ):
        if not (0 <= label_idx < 38):
            label_idx = 31

        self.feature_inds = feature_inds
        self.label_idx = label_idx

        # Open the HDF5 file to access data and indices
        self.h5_file = h5py.File(hd_file, "r")
        self.volumes = self.h5_file["volumes"]
        self.data = self.volumes["capped_data" if use_capped else "data"]

        # Set appropriate indices based on dataset type
        if type == "train":
            self.indices = self.volumes["train_idx"][:]
        elif type == "validation":
            self.indices = self.volumes["val_idx"][:]
        else:  # test
            self.indices = self.volumes["test_idx"][:]

        # Get case IDs for the dataset
        all_case_ids = [id.decode("utf-8") for id in self.volumes["case_ids"][:]]
        self.cases = [all_case_ids[i] for i in self.indices]

        if normalize or normalize_output:
            # Get statistics from HDF5 instead of JSON
            stats_group = self.h5_file.get("capped_statistics")
            if stats_group is None:
                raise ValueError(
                    "Statistics not found in HDF5 file. Run store_statistics() first."
                )

            all_means = stats_group["means"][:]
            all_stds = stats_group["stds"][:]

            self.target_mean = all_means[self.label_idx]
            self.target_std = all_stds[self.label_idx]

            if self.feature_inds:
                self.mean = [all_means[i] for i in self.feature_inds]
                self.std = [all_stds[i] for i in self.feature_inds]
            else:
                # Otherwise use all means/stds except the label index
                self.mean = np.delete(all_means, self.label_idx).tolist()
                self.std = np.delete(all_stds, self.label_idx).tolist()

        transform_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
        ]
        if normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.transform = transforms.Compose(transform_list)

        target_transforms = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
        ]

        if normalize_output:
            target_transforms.append(
                transforms.Normalize(mean=self.target_mean, std=self.target_std),
            )

        self.target_transform = transforms.Compose(target_transforms)

    def __del__(self):
        """Clean up resources when object is garbage collected."""
        if hasattr(self, "h5_file") and self.h5_file is not None:
            self.h5_file.close()

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        # Get the actual index in the HDF5 dataset
        data_idx = self.indices[idx]

        # Extract data from the HDF5 file
        # The data is stored as [samples, height, width, channels]
        sample_data = self.data[data_idx]

        # Extract label and input channels
        label = sample_data[:, :, self.label_idx]

        if self.feature_inds:
            # Extract only the requested feature channels
            image = sample_data[:, :, self.feature_inds]
        else:
            # Use all channels except the label channel
            image = np.delete(sample_data, self.label_idx, axis=2)

        # Apply transformations
        image = self.transform(image)
        label = self.target_transform(label)

        return image, label

    def get_case_from_index(self, idx):
        return self.cases[idx]


class H5IMCDataset(Dataset):
    """
    Generic dataset for `data.h5`.
    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    split : {'train', 'val', 'test', 'all'}
        Which subset to expose.
    in_channels : list[int] or slice or None
        Channels to use as network input (default: *all*).
    target_channel : int or None
        If given, that channel is returned as the target (present/absent mask);
        otherwise the full tensor is returned.
    transform : callable or None
        Optional torchvision-style transform applied to the **input tensor only**.
    joint_transform : callable or None
        Optional torchvision-style transform applied to the **input and target tensors simultaneously**.
        This is useful for spatial augmentations like flips and rotations.
    exclude_target_from_input : bool
        If True and `target_channel` is set, the target channel will be excluded from the input tensor.
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        in_channels=None,
        target_channel: int | None = None,
        transform=None,
        joint_transform=None,  # Add new argument
        exclude_target_from_input=True,  
        binarize_target=True,
        binarize_threshold=0
    ):
        self.h5_path = Path(h5_path)
        self.split = split
        self.in_channels = in_channels
        self.target_channel = target_channel
        self.transform = transform
        self.joint_transform = joint_transform  # Store new argument
        self.exclude_target_from_input = exclude_target_from_input
        self.binarize_target = binarize_target
        self.threshold = binarize_threshold

        # ----- read small metadata once (no heavy data) -----
        with h5py.File(self.h5_path, "r") as f:
            v = f["/volumes"]
            if split == "train":
                self.indices = v["train_idx"][:]
            elif split == "val":
                self.indices = v["val_idx"][:]
            elif split == "test":
                self.indices = v["test_idx"][:]
            elif split == "all":
                self.indices = np.arange(len(v["case_ids"]))
            else:
                raise ValueError(f"Unknown split: {split}")

            all_case_ids = [cid.decode("utf-8") for cid in v["case_ids"][:]]
            self.cases = [all_case_ids[i] for i in self.indices]

            # statistics are handy for normalisation
            if "/statistics" in f:
                self.means = torch.tensor(
                    f["/statistics/means"][:], dtype=torch.float32
                )
                self.stds = torch.tensor(f["/statistics/stds"][:], dtype=torch.float32)
            else:
                self.means = self.stds = None

        # Lazy file handle (one per process)
        self._h5 = None

    # ---------- Python special methods ----------
    def __len__(self):
        return len(self.indices)

    def __del__(self):
        self.close()

    # ---------- public helpers ----------
    def get_case_from_index(self, idx: int) -> str:
        """Returns the case ID for a given dataset index."""
        return self.cases[idx]

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # ---------- core ----------
    def _ensure_open(self):
        """Open the HDF5 file the first time we need it *in this process*."""
        if self._h5 is None:
            # SWMR=True allows safe concurrent *reading* by many processes
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)

    def __getitem__(self, idx):
        self._ensure_open()

        real_idx = self.indices[idx]
        vol = self._h5["/volumes/data"][real_idx][1:-1, 1:-1, :] # (H, W, C) numpy
        vol = torch.from_numpy(vol).permute(2, 0, 1).float()  # (C, H, W)

        # channel selection ---------------------------------------------------
        if self.in_channels is not None:
            x = vol[self.in_channels]  # choose network inputs
        else:
            # If we have a target channel and want to exclude it from input
            if self.target_channel is not None and self.exclude_target_from_input:
                # Use all channels except the target
                all_channels = list(range(vol.shape[0]))
                input_channels = [ch for ch in all_channels if ch != self.target_channel]
                x = vol[input_channels]
            else:
                x = vol

        # normalise -----------------------------------------------------------
        if self.means is not None:
            if self.in_channels is not None:
                means = self.means.view(-1, 1, 1)[self.in_channels]
                stds = self.stds.view(-1, 1, 1)[self.in_channels]
            elif self.target_channel is not None and self.exclude_target_from_input:
                # Exclude target channel from normalization stats
                all_channels = list(range(len(self.means)))
                input_channels = [ch for ch in all_channels if ch != self.target_channel]
                means = self.means.view(-1, 1, 1)[input_channels]
                stds = self.stds.view(-1, 1, 1)[input_channels]
            else:
                means = self.means.view(-1, 1, 1)
                stds = self.stds.view(-1, 1, 1)

            x = (x - means) / (stds + 1e-6)

        # build target --------------------------------------------------------
        if self.target_channel is not None:
            y = vol[self.target_channel]  # (H, W) raw signal

            if self.binarize_target:
                y = (y > self.threshold).to(dtype=torch.float32)
            else:
                y = y.to(dtype=torch.float32)
            y = y.unsqueeze(0) # Add channel dimension -> (1, H, W)            # Apply joint transforms to both image and mask simultaneously

            if self.joint_transform:
                # Stack image and mask, apply transform, then unstack
                stacked = torch.cat((x, y), dim=0)
                transformed_stacked = self.joint_transform(stacked)
                x = transformed_stacked[:-1, :, :]
                y = transformed_stacked[-1:, :, :]

                if self.transform:
                    x = self.transform(x)

            return x, y
        else:
            if self.transform:
                x = self.transform(x)
            return x


if __name__ == "__main__":
    # Test the H5IMCDataset
    try:
        from torch.utils.data import DataLoader
        import time
        
        # Test with default parameters
        dataset = H5IMCDataset("data.h5", split="train", target_channel=31)
        print(f"Dataset length: {len(dataset)}")
        
        # Test getting a sample
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Input dtype: {x.dtype}")
            print(f"Target dtype: {y.dtype}")
        
        # Test without target channel
        dataset_no_target = H5IMCDataset("data.h5", split="train")
        if len(dataset_no_target) > 0:
            x_only = dataset_no_target[0]
            print(f"Input only shape: {x_only.shape}")
        
        # Test with specific input channels
        dataset_subset = H5IMCDataset("data.h5", split="train", in_channels=[0, 1, 2], target_channel=31)
        if len(dataset_subset) > 0:
            x_sub, y_sub = dataset_subset[0]
            print(f"Subset input shape: {x_sub.shape}")
        
        # Test with concurrent workers (multiprocessing)
        print("\nTesting with DataLoader and multiple workers...")
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            num_workers=2, 
            shuffle=True,
            pin_memory=True
        )
        
        # Time a few batches to check performance
        start_time = time.time()
        for i, (batch_x, batch_y) in enumerate(dataloader):
            print(f"Batch {i}: Input {batch_x.shape}, Target {batch_y.shape}")
            if i >= 2:  # Test first 3 batches
                break
        
        elapsed = time.time() - start_time
        print(f"Processed 3 batches in {elapsed:.2f} seconds")
        
        # Test single worker for comparison
        print("\nTesting with single worker...")
        dataloader_single = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,  # Single threaded
            shuffle=True
        )
        
        start_time = time.time()
        for i, (batch_x, batch_y) in enumerate(dataloader_single):
            if i >= 2:
                break
        
        elapsed_single = time.time() - start_time
        print(f"Single worker processed 3 batches in {elapsed_single:.2f} seconds")
        
        print("H5IMCDataset test completed successfully!")
        
    except FileNotFoundError:
        print("data.h5 file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error testing H5IMCDataset: {e}")
        import traceback
        traceback.print_exc()
