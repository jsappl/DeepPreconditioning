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


def _select_subset(source: Path, min: int, max: int):
    """Generartor for files based on degrees of freedom."""
    files = list(source.glob("*.npz"))

    for file in tqdm(files):
        dof = _get_dof(file)

        if min < dof < max:
            yield file


def main() -> None:
    """The main subset generation loop."""
    source = ROOT / "stand_small_train"
    files = _select_subset(source, min=2950, max=3050)

    for n_files, file in enumerate(files):
        if n_files < 900:
            os.link(file, ROOT.parent / "processed/stand_small_train" / file.name)
        elif n_files < 1000:
            os.link(file, ROOT.parent / "processed/stand_small_test" / file.name)
        elif n_files == 1000:
            print(f"Selected {n_files} files, stopping.")
            break
        else:
            raise ValueError(f"Unexpected number of files: {n_files}. Expected less than 1000.")


if __name__ == "__main__":
    main()
