To understand how to use Singularity with GPU enabled, refer to the following documentation:

- [NVIDIA Driver Installation Quickstart](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Driver_Installation_Quickstart.pdf)
- [CUDA Toolkit Documentation v10.2.89](https://docs.nvidia.com/cuda/archive/10.2/)
- [Singularity GPU Support](https://sylabs.io/guides/3.6/user-guide/gpu.html)

## ./eeg_dementia_classification_2023_0905.def
This is the Singularity definition file for eeg_dementia_classification_2023_0905.sif.

## ./eeg_dementia_classification_2023_0905.sif
This Singularity image file is designed to run our Python scripts.

## ./singularity-aliases.bash
This file contains bash aliases that simplify the use of Singularity.

## How to use singularity-aliases.bash:
Place this file in $HOME as ~/singularity-aliases.bash.
Add the following line to your .bashrc or .bash_profile:
```
. $HOME/singularity-aliases.bash # This will automatically load the aliases.

```
Store your .sif file or sandbox directory under the singularity directory of your project:
```
./singularity
├── YOUR_REAL_SIF_FILE.sif
├── YOUR_REAL_SANDBOX_DIR
├── image.sif -> YOUR_REAL_SIF_FILE.sif (symlink)
└── image -> YOUR_REAL_SANDBOX_DIR (symlink)
```

``` bash
Usage Examples:

.sif files: Use commands like $ sshell, $ sipy, $ spy script.py, and $ sjpy.
Writable sandbox: Use the same commands with a w suffix (e.g., $ sshellw).
Building: Use $ sbuild example.def and its variants.```
```
