"""Generate linear systems from OpenFOAM based on different sludge patterns."""

import shutil
import subprocess
from pathlib import Path

import dvc.api
import numpy as np
import triangle as tr
from scipy.sparse import coo_matrix, save_npz
from stl import mesh

ROOT: Path = Path(__file__).parents[2]


def _sludge_pattern(resolution: int) -> None:
    """Generate a random sludge pattern at the bottom of the tank.

    The resulting mesh is saved as an STL file in the correct OpenFOAM directory.

    Args:
        resolution: The resolution of the OpenFOAM mesh.
    """
    positions_x = np.linspace(1, 25, num=resolution)
    positions_y = .0625 * positions_x - 6.0625
    positions_y[1:-1] += np.random.normal(loc=.25, scale=.1, size=resolution - 2)

    vertices = np.zeros((2 * resolution, 3))
    vertices[:, 0] = np.concatenate((positions_x, positions_x[::-1]))
    vertices[:, 1] = np.concatenate((positions_y, positions_y[::-1]))
    vertices[resolution:, 2] = resolution * [-.5]

    vertice_ids = np.array(range(2 * resolution))
    triangles = tr.triangulate(
        dict(
            vertices=vertices[:, [0, 2]],
            segments=np.stack((vertice_ids, (vertice_ids + 1) % len(vertice_ids))).T,
        ),
        "pq",
    )
    faces = triangles["triangles"]

    sludge = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for face_index, face in enumerate(faces):
        for dimension in range(3):
            sludge.vectors[face_index][dimension] = vertices[face[dimension], :]

    stl_directory = ROOT / "foam/sim/constant/triSurface/"
    stl_directory.mkdir(parents=True, exist_ok=True)
    sludge.save(stl_directory / "sludge.stl")


def _build_matrix(csv_file: Path) -> coo_matrix:
    """Build an SPD matrix from a csv file.

    Args:
        csv_file: Path to the matrix csv file from OpenFOAM.

    Returns:
        matrix: The matrix in COO format.

    Raises:
        AssertionError: If the matrix is not positive definite.
    """
    data = np.genfromtxt(csv_file, delimiter=",")

    row = data[:, 0]
    col = data[:, 1]
    val = -data[:, 2]

    n_rows = int(max(row)) + 1
    matrix = coo_matrix((val, (row, col)), shape=(n_rows, n_rows))

    assert ((matrix.transpose() != matrix).nnz == 0), "Generated matrix is non-symmetric matrix"

    eigenvalues = np.linalg.eigvals(matrix.toarray())
    assert np.all(eigenvalues > 0), "Generated matrix is not positive definite"

    return matrix


def main() -> None:
    """Simulate random sluge patterns."""
    np.random.seed(69420)

    params = dvc.api.params_show()

    for index in range(params["number_samples"]):
        _sludge_pattern(resolution=params["resolution"])

        case_directory = ROOT / f"assets/data/raw/sludge_patterns/case_{index:04}/"
        case_directory.mkdir(parents=True, exist_ok=True)

        # Run simulation and dump matrix.
        command = "docker exec openfoam /bin/bash -i -c 'cd sim/ && ./Allrun'"
        subprocess.run(command, cwd=ROOT / "foam", shell=True)

        matrix = _build_matrix(ROOT / "foam/sim/matrix.csv")
        save_npz(case_directory / "matrix.npz", matrix, compressed=False)

        for type_ in ["solution", "right_hand_side"]:
            shutil.copy(ROOT / f"foam/sim/{type_}.csv", case_directory / f"{type_}.csv")


if __name__ == "__main__":
    main()
