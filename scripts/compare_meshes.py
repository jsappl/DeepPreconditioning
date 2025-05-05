"""Compare generalization capabilities on different meshes."""

import subprocess
from pathlib import Path

import dvc.api
import numpy as np
import torch
from scipy.sparse import load_npz, save_npz
from spconv.pytorch import SparseConvTensor

import uibk.deep_preconditioning.generate_data as generate_data
import uibk.deep_preconditioning.metrics as metrics
import uibk.deep_preconditioning.model as models

ROOT: Path = Path(__file__).parents[1]
RESULTS_DIRECTORY: Path = Path("./assets/results/")
MESH_CELLS: list[int] = [2, 3, 4, 5, 6]

rng = np.random.default_rng(seed=69)


def generate_mesh(mesh_cells: int, resolution: int) -> None:
    """Generate a meshe for some given resolution."""
    generate_data._sludge_pattern(resolution=resolution)

    case_directory = ROOT / "assets/data/meshes/"
    case_directory.mkdir(parents=True, exist_ok=True)

    # Run simulation and dump matrix.
    command = "docker exec openfoam /bin/bash -i -c 'cd foam/sim/ && ./Allrun'"
    subprocess.run(command, cwd=ROOT / "foam", shell=True)

    matrix = generate_data._build_matrix(ROOT / "foam/sim/matrix.csv")

    save_npz(case_directory / f"mesh_cells_{mesh_cells}_matrix.npz", matrix, compressed=False)


def main():
    """Generate different meshes for the generalization test."""
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    torch.manual_seed(69)

    params = dvc.api.params_show()

    for mesh_cells in MESH_CELLS:
        pattern = f"s/^res [0-9]+/res {mesh_cells}/"
        subprocess.run(f"sed -i -E '{pattern}' blockMeshDict", cwd=ROOT / "foam/sim/system", shell=True)
        generate_mesh(mesh_cells, params["resolution"])

    model = getattr(models, params["model"])(params["channels"])
    model.load_state_dict(torch.load(Path("./assets/checkpoints/best.pt"), weights_only=True))
    model = model.to(device)

    with (RESULTS_DIRECTORY / "compare_meshes.csv").open(mode="w") as file_io:
        file_io.write("mesh_cells,size,kappa_pre,kappa_post\n")

        for mesh_cells in MESH_CELLS:
            matrix = load_npz(ROOT / f"assets/data/meshes/mesh_cells_{mesh_cells}_matrix.npz").toarray()
            condition_number_pre = np.linalg.cond(matrix).item()
            matrix = SparseConvTensor.from_dense(
                torch.tril(torch.tensor(matrix, dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(-1)
            )
            condition_number_post = metrics.condition_loss(matrix, model(matrix)).item()
            line = f"{mesh_cells},{matrix.spatial_shape[0]},{condition_number_pre:.0f},{condition_number_post:.0f}"
            file_io.write(line + "\n")


if __name__ == "__main__":
    main()
