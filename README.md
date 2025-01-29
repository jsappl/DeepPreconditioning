<div align="center">

# Deep Preconditioning

Design preconditioners with a CNN to accelerate the conjugate gradient method.

[![python](https://img.shields.io/badge/python-%3E%3D3.12-blue?logo=python)]()
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

## Setup (Linux)

This has been tested with

- Debian 10.2.1-6 (GNU/Linux 6.1.0-28-amd64 x86_64)
- Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz
- NVIDIA GeForce RTX 3070
- NVIDIA Driver Version 550.54.14
- CUDA Version 12.4
- Python 3.12.8

Clone this repository and `cd` into the project root.

```bash
git clone git@github.com:jsappl/DeepPreconditioning.git
cd DeepPreconditioning/
```

We use [PDM](https://pdm-project.org/en/latest/) to build the package and manage dependencies so make sure it is installed.
After selecting a Python interpreter, PDM will ask you whether you want to create a virtual environment for the project.
Having one is optional but highly recommended.
PDM will try to auto-detect possible virtual environments.
Run `pdm install` to install dependencies from the `pdm.lock` file and restore the project environment.

## Generating Data Set

The data set of linear systems resulting from the discretization of the pressure correction equation is generated using OpenFOAM 7.
The most convenient way to use OpenFOAM 7 is to download and run the official Docker container.
For further instructions please visit <https://openfoam.org/download/7-linux/>.
First, compile our custom `newInterFoam` solver inside the container.

```bash
openfoam7-linux  # run in root folder
cd foam/newInterFoam/  # inside Docker container
wmake
```

After compiling the solver keep the Docker container up and running.
In another shell run the `generate` stage of the `dvc` pipeline.
The data set is automatically generated making use of the OpenFOAM 7 Docker container.

## Developing

Use PDM to add another dependency and update the projects `pdm.lock` file afterward.

```bash
pdm add <some-dependency>
pdm update
```

Run `pdm sync --clean` to remove packages that are no longer in the `pdm.lock` file.

Version numbers are (roughly) assigned and incremented according to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Write commit messages according to the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) convention.
Keep a changelog and stick to this style guide <https://common-changelog.org/>.
Use these tools for code formatting and linting.

- `ruff` (includes `isort` and `flake8`)
- `yapf`

They are automatically configured by the `pyproject.toml` file.
