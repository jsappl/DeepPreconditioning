"""Select a small subset of Kaggle StAnD (Small problems)."""

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT: Path = Path(__file__).parents[1] / "./assets/data/raw/"


def _get_dof(file: Path) -> int:
    """Open a file and return degrees of freedom."""
    assert file.is_file(), f"{file} is not a file"
    _, _, solution, _ = np.load(file).values()
    return len(solution)


def _select_subset(source: Path, destination: Path, min: int, max: int, max_files: int) -> None:
    """Select a subset of files based on degrees of freedom."""
    files = list(source.glob("*.npz"))
    count = 0

    for file in tqdm(files):
        dof = _get_dof(file)

        if min < dof < max:
            os.link(file, destination / file.name)
            count += 1

        if count >= max_files:
            break


def main() -> None:
    """The main subset generation loop."""
    for mode in ["train", "test"]:
        source = ROOT / f"stand_small_{mode}"
        destination = ROOT.parent / f"processed/stand_small_{mode}"
        destination.mkdir(parents=True, exist_ok=True)

        _select_subset(source, destination, min=3950, max=4050, max_files=100 if mode == "train" else 10)


if __name__ == "__main__":
    main()
