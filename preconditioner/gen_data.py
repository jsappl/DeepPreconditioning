"""Generate OpenFOAM system matrices based on baffles or sludge patterns."""
import subprocess
from itertools import product

import numpy as np
import triangle as tr
from scipy.sparse import save_npz
from stl import mesh

from config import CFG
from utils import is_positive_definite


def _sludge_pattern(resolution=128):
    """Random sludge pattern at bottom of tank."""
    x_pos = np.linspace(1, 25, num=resolution)
    y_pos = .0625*x_pos-6.0625
    y_pos[1:-1] += np.random.normal(loc=.25, scale=.1, size=resolution-2)

    vertices = np.zeros((2*resolution, 3))
    vertices[:, 0] = np.concatenate((x_pos, x_pos[::-1]))
    vertices[:, 1] = np.concatenate((y_pos, y_pos[::-1]))
    vertices[resolution:, 2] = resolution*[-.5]

    vert_id = np.array(range(2*resolution))
    triags = tr.triangulate(dict(vertices=vertices[:, [0, 2]],
                                 segments=np.stack((vert_id, (vert_id+1)
                                                    % len(vert_id))).T), 'pq')
    faces = triags['triangles']

    sludge = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            sludge.vectors[i][j] = vertices[face[j], :]

    sludge.save('../foam/sim/constant/triSurface/sludge.stl')
    return True


def _baffle():
    """Simulating baffles with different height and position in tank."""
    parameters = dict(
        xmin=[2., 2.5, 3., 3.5, 4., 4.5],
        height=[1., 1.5, 2., 2.5, 3.]
    )

    param_values = [v for v in parameters.values()]

    for idx, (xmin, height) in enumerate(product(*param_values)):
        # Open Dict template.
        with open('../foam/sim/system/snappyHexMeshDict.org', 'r') as file_:
            data = file_.readlines()

        # Baffle min point.
        data[26] = data[26].replace('xx', str(xmin))\
            .replace('yy', '-'+str(height))\
            .replace('zz', '-0.25')
        # Baffle max point.
        data[27] = data[27].replace('xx', str(xmin+.5))\
            .replace('yy', '0')\
            .replace('zz', '0')

        # Write actual Dict file used for simulation.
        with open('../foam/sim/system/snappyHexMeshDict', 'w') as file_:
            file_.writelines(data)

        # Run simulation and dump matrix.
        subprocess.call(['../foam/sim/Allrun'])
        l_matrix = is_positive_definite('L.csv')
        save_npz('./data/L'+str(idx).zfill(3)+'.npz', l_matrix)
    return True


def _sludge():
    """Simulating random sluge patterns."""
    np.random.seed(CFG['SEED'])
    for idx in range(CFG['DATA_COUNT']):
        _sludge_pattern()

        # Run simulation and dump matrix.
        subprocess.call(['../foam/sim/Allrun'])
        l_matrix = is_positive_definite('L.csv')
        save_npz('./data/L'+str(idx).zfill(3)+'.npz', l_matrix)
    return True


def main():
    # _baffle()
    _sludge()
    return True


if __name__ == '__main__':
    main()
