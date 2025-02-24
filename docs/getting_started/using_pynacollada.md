---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Using Pynacollada

Pynacollada is an open-source software package to provide detailed and accessible tutorials using curated data sets in Neuroscience. 

## Installation

### Quick install
Clone the repository

```shell
git clone https://github.com/neurostatslab/pynacollada.git
```

Navigate to the repository and install

```shell
pip install .
```

Build all Jupyter notebooks

```shell
make all
```

Check available Jupyter notebooks to build

```shell
make help
```

Build specific Jupyter notebook (e.g. "place_cells")

```shell
make place_cells
```

### Detailed instructions
:::::::{tab-set}
:sync-group: category



::::::{tab-item} conda / miniforge
:sync: conda

:::{warning}

Due to [recent changes](https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research) to Anaconda's Terms of Service, the Simons Foundation cannot use the `defaults` conda channel and it is blocked on all Flatiron Institute wireless networks. You need to specify `conda-forge` instead (which is community-managed and open to all). The following instructions do so, but if you follow your normal workflow, you may hit issues.

:::

1. Install [miniforge](https://github.com/conda-forge/miniforge) if you do not have some version of `conda` or `mamba` installed already.
2. Create the new virtual environment by running:
    ```shell
    conda create --name ccn-jan25 pip python=3.11 -c conda-forge
    ```
    Note the `-c conda-forge`!

3. Activate your new environment and navigate to the cloned repo: 
    ```shell
    conda activate ccn-jan25
    cd ccn-software-jan-2025
    ```
::::::

::::::{tab-item} uv
:sync: uv

:::::{tab-set}
:sync-group: os

::::{tab-item} Mac/Linux
:sync: posix

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) by running:
   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
2. Restart your terminal to make sure `uv` is available.
3. Install python 3.11:
   ```shell
   uv python install 3.11
   ```
   
4. Navigate to your cloned repo and create a new virtual environment:
   ```shell
   cd ccn-software-jan-2025
   uv venv -p 3.11
   ```
   
5. Activate your new virtual environment by running:
   ```shell
   source .venv/bin/activate
   ```
::::

::::{tab-item} Windows
:sync: windows

Open up `powershell`, then:

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Install python 3.11:
   ```powershell
   uv python install 3.11
   ```
   
3. Navigate to your cloned repo and create a new virtual environment:
   ```powershell
   cd ccn-software-jan-2025
   uv venv -p 3.11
   ```
   
4. Activate your new virtual environment by running:
   ```powershell
   .venv\Scripts\activate
   ```

   :::{warning}
   You may receive an error saying "running scripts is disabled on this system". If so, run `Set-ExecutionPolicy -Scope CurrentUser` and enter `Unrestricted`, then press `Y`.
   
   You may have to do this every time you open powershell.
   
   :::

::::
:::::
::::::

:::::::

## User API

Import the module

```{code-cell} ipython3
import pynacollada as nac
```

### Downloading data sets

Pynacollada provides a single interface to download and load provided data sets by their string identifier. Current data sets:

| Name | Data Set |
| ---- | -------- |
| "mesoscale_activity" | Link | 
| "place_cells" | Link | 

You can load in the data using the function 

```{code-cell} ipython3
data = nac.load_data("place_cells")
print(data)
```

This will first download the data if it does not exist in `nac.config["data_dir"]`


### Configuration settings
Check configuration

```{code-cell} ipython3
nac.config
```

Update configuration

```{code-cell} ipython3
nac.config["data_dir"] = "/path/to/data"
nac.config
```

Save configuration to current directory

```python
nac.config.save()
```

Save config to different directory
```python
nac.config.save("/path/to/config")
```

Pynacollada will load in config file from current directory. To load from a different file

```python
nac.config.load("/path/to/config")
```

