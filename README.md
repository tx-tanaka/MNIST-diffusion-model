# MNIST-diffusion-model

Reverse-time diffusion on MNIST: train a time-conditioned U-Net to predict Gaussian
noise (equivalently, the score), then generate synthetic handwritten digits by
integrating the reverse-time SDE with Euler-Maruyama.

Both notebooks run on an NVIDIA GPU or on CPU. They detect the hardware at startup
(`torch.cuda.is_available()`) and shrink the training configuration when no GPU is
found, so a CPU-only machine still completes the full workflow in roughly ten
minutes -- with correspondingly rougher digits.

| Notebook | Contents |
| --- | --- |
| `reverse_time_diffusion_MNIST_homework.ipynb` | Parts A-D: forward OU diffusion, denoising score matching, U-Net training, reverse-time sampling |
| `hw.ipynb` | Working notebook for the same material |

## Requirements

- **Python 3.10 - 3.14** (developed on 3.14)
- **~2.5 GB of disk space** for a CUDA build of PyTorch, or ~400 MB for the CPU build
- **Optional: an NVIDIA GPU** with an up-to-date driver
- An internet connection for the first run, which downloads MNIST (~55 MB) into `data/`

## Installation

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch for your hardware

PyTorch publishes a separate build per compute backend, so this step -- and only
this step -- depends on your machine. Pick the single row that matches it:

| Hardware | Command |
| --- | --- |
| NVIDIA GPU, driver supporting CUDA 13 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| NVIDIA GPU, older driver (CUDA 12.6) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| No GPU / CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| AMD GPU (ROCm, Linux only) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1` |
| Apple Silicon (M-series) | `pip install torch torchvision` |

**Which CUDA version?** Run `nvidia-smi`. The `CUDA Version` shown in the header is
the highest your driver supports -- choose an index at or below it. If `nvidia-smi`
is not found, you have no usable NVIDIA GPU: use the CPU row.

**Windows users, note:** a plain `pip install torch` pulls the CPU-only wheel from
PyPI on Windows. The `--index-url` flag is what gets you CUDA. (On Linux the default
PyPI wheel is already CUDA-enabled; on macOS it is CPU/MPS.)

The interactive selector covering every other combination lives at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A CUDA build prints a version tagged with the backend and `True`, for example
`2.14.1+cu130 True`. A CPU build prints `2.14.1+cpu False`, which is expected and
fine -- the notebooks will simply use the smaller CPU configuration.

If it prints `False` on a machine that does have an NVIDIA GPU:

- Check the version string. Ending in `+cpu` means the CPU wheel was installed;
  reinstall with `pip install --force-reinstall torch torchvision --index-url <CUDA index above>`.
- If you installed PyTorch *after* starting Jupyter, **restart the kernel**. A running
  kernel keeps the `torch` it imported at startup, so a later `pip install` has no
  effect until the process restarts.
- Update your NVIDIA driver if it predates the CUDA version you selected.

## Running the notebooks

```bash
jupyter lab
```

Open either notebook, confirm the selected kernel is the `.venv` you created, and
run all cells. The first cell prints the detected device and the configuration it
chose; generated images and the model checkpoint are written to `diffusion_outputs/`.

On Apple Silicon the notebooks fall back to CPU, because they test only for CUDA.
To use the GPU there, change the device line to prefer
`torch.device("mps") if torch.backends.mps.is_available() else ...`.

