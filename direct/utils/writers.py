# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import pathlib
from typing import Callable, DefaultDict, Dict, Optional, Union

import h5py  # type: ignore
import numpy as np
import torch

logger = logging.getLogger(__name__)


def write_output_to_h5(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
) -> None:
    """Write dictionary with keys filenames and values torch tensors to h5 files.

    Parameters
    ----------
    output: dict
        Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
        where num_channels is typically 1 for MRI.
    output_directory: pathlib.Path
    volume_processing_func: callable
        Function which postprocesses the volume array before saving.
    output_key: str
        Name of key to save the output to.
    create_dirs_if_needed: bool
        If true, the output directory and all its parents will be created.

    Notes
    -----
    Currently only num_channels = 1 is supported. If you run this function with more channels the first one
    will be used.
    """
    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)

    for idx, (volume, extra_data, filename) in enumerate(output):
        # The output has shape (slice, 1, height, width)
        if isinstance(filename, pathlib.PosixPath):
            filename = filename.name

        logger.info(f"({idx + 1}/{len(output)}): Writing {output_directory / filename}...")

        reconstruction = volume.numpy()[:, 0, ...].astype(np.float32)

        if volume_processing_func:
            reconstruction = volume_processing_func(reconstruction)

        # Smart reshaping based on tvec
        if extra_data and "tvec" in extra_data:
            tvec = extra_data["tvec"]
            if isinstance(tvec, torch.Tensor):
                tvec_shape = tvec.shape
            else:
                tvec_shape = tvec.shape
            
            # tvec shape is typically (Batch, Slices, Time) e.g. (1, 5, 9)
            # reconstruction shape is (Total_Slices, H, W) e.g. (45, 512, 144)
            if len(tvec_shape) == 3:
                slices, time = tvec_shape[1], tvec_shape[2]
                if reconstruction.shape[0] == slices * time:
                    logger.info(f"Reshaping reconstruction from {reconstruction.shape} to ({slices}, {time}, ...)")
                    # Dataset iterates Time-Major (Time, Slice), so reshape to (Time, Slice) first
                    reconstruction = reconstruction.reshape(time, slices, reconstruction.shape[1], reconstruction.shape[2])
                    # Then transpose to (Slice, Time) to match tvec
                    reconstruction = reconstruction.transpose(1, 0, 2, 3)

        with h5py.File(output_directory / filename, "w") as f:
            f.create_dataset(output_key, data=reconstruction)
            if extra_data:
                logger.info(f"Writing extra data keys: {list(extra_data.keys())}")
                for key, value in extra_data.items():
                    if key == output_key:
                        continue
                    
                    if isinstance(value, torch.Tensor):
                        value = value.numpy()
                    
                    # Squeeze singleton batch dimension if present
                    if value.ndim > 0 and value.shape[0] == 1:
                        value = value.squeeze(0)

                    try:
                        f.create_dataset(key, data=value)
                    except Exception as e:
                        logger.error(f"Failed to write key {key}: {e}")
            else:
                logger.warning(f"No extra data found for {filename}")
