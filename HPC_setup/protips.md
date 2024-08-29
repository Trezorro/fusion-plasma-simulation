

# TIL: Running Python with CUDA on a Cluster Using SLURM

## Key Insights:
- **Main Node vs. Compute Nodes**:
  - **Main Node (datamininghpc)**: This is the default node you log into when accessing the cluster. It **does not** have CUDA-enabled GPUs, so any CUDA-related code will not run here if you run it directly. It ONLY works via SLURM job submission.
  - **Compute Nodes (e.g., calcium, zirconium)**: These nodes are equipped with GPUs. CUDA-enabled code will run successfully on these nodes.


## Setting Up Python with CUDA:
- **Miniconda Installation**: Direct installation of Python with tar and pip is challenging due to missing OpenSSL versions. Instead, use Miniconda:
  ```bash
  conda create -n py11 python=3.11.9 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

  This environment is compatible with CUDA 11, which is installed on the compute nodes. If you encounter issues, it may help to install:

  ```bash
  conda install cudnn
  ```

## Submitting Jobs with SLURM:
- **Test Scripts**: Use a bash script (`testcuda.sh`) and a Python script (`testcuda.py`) to verify that your environment correctly utilizes GPU resources. These scripts will help ensure CUDA is accessible, and the environment is set up properly.

## Pro Tips:
- **Quick File Creation from shell**: To quickly create or overwrite files via SSH, use `cat > filename.txt`, paste your clipboard content, and press `Ctrl+D` to save.
- **Job Submission**: Submit jobs using SLURM and verify output with `cat slurm-<jobid>.out` to check job status and results.
- Create a shortcut for logging into the cluster by adding the following to your `~/.ssh/config` file:
  ```
  Host HPC
      HostName datamininghpc.win.tue.nl
      User TUE_usernumber
  ```
  Now you can log in using `ssh TUE` instead of the full command. Setup passwordless SSH Login for faster access using SSH keys.


### Debugging CUDA Availability
  You can test if CUDA is available in two ways. Using SLURM or by logging in to nodes directly. 

  1. **Submit a SLURM Job with GPU Request**:
     I created a simple script `testcuda.sh` that runs a Python script `testcuda.py` to check if CUDA is available. These serve as a nice start for any job submission, to make debugging easier. I attached the scripts below for reference. After saving them on the server, run:
     ```bash
     sbatch -p zirconium --gres=gpu:1 testcuda.sh
     ```
     To ensure that CUDA is made available, you must submit a SLURM job with the `--gres=gpu:1` option. The `--gres=gpu:1` option requests one GPU for your job. Adjust the number if you need more GPUs.


  2. **Testing CUDA Availability directly in the shell**:
   
        _**Note**: It should not be strictly necessary to log into these nodes directly to test CUDA; submitting a SLURM job will automatically allocate a GPU-enabled node if requested._
     
     If you prefer to test CUDA directly, you can SSH into a compute node:

        ```bash
        ssh TUE_usernumber@datamininghpc.win.tue.nl  # Main Node node
        ssh TUE_usernumber@calcium.win.tue.nl    # Example for calcium node
        ```
     After logging in, you can check CUDA availability by running:
     ```bash
     . .bashrc            # Activate base profile and make conda available if necessary
     conda activate py11  # Activate your environment
     python               # Enter Python shell
     ```
     Inside Python:
     ```python
     import torch
     print(torch.cuda.is_available())  # Check if CUDA is available
     ```
        > Running the above code on `datamininghpc` will return `False`, indicating no CUDA support. On nodes like `calcium` or `zirconium`, it will return `True`, indicating CUDA is available.




