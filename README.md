<div align="center">

# Deep Preconditioning

Design preconditioners with a CNN to accelerate the conjugate gradient method.

[![python](https://img.shields.io/badge/python-%3E%3D3.11-blue?logo=python)]()
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

## Setup (Linux)

This has been tested with

* Debian 10.2.1-6 (GNU/Linux 5.10.0-28-amd64 x86_64)
* Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz
* NVIDIA GeForce RTX 3070
* NVIDIA Driver Version 550.54.15
* CUDA Version 12.4
* Python 3.11.9

Clone this repository and `cd` into the project root.

```bash
git clone git@github.com:jsappl/DeepPreconditioning.git
cd DeepPreconditioning/
```

We use [PDM](https://pdm-project.org/en/stable/) to build the package and manage dependencies so make sure it is installed.
After selecting a Python interpreter, PDM will ask you whether you want to create a virtual environment for the project.
Having one is optional but highly recommended.
PDM will try to auto-detect possible virtual environments.
Run `pdm install` to install dependencies from the `pdm.lock` file and restore the project environment.

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
