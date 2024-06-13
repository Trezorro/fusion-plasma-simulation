WARNING $SELECTION_PLACEHOLDER$
BOTH THE CLUSTER AND THIS WIKI ARE STILL IN ACTIVE DEVELOPMENT. EXPECT MAINTENANCE DOWNTIME AND SETUP CHANGES. 

Gaining Access 
The Data Mining group has its own compute cluster that is available to group members and, under certain circumstances, to students. If you have received access, you can SSH to datamininghpc.win.tue.nl to access it (on Windows, you can use PuTTY for this purpose). Log in with the username TUE_<username>, where <username> is your TU/e account, and use your TU/e password.  
ssh TUE_<username>@datamininghpc.win.tue.nl  

To request access, please contact one of the administrators. We will require the following information: 
- Your name 
- Your TU/e account username 
- The reason you need access to the cluster (students only) 
- The duration for which you need access (students only) 

If a student needs access for an internship or graduation project, their supervisor should request access for the student, or the student must show that their supervisor has given permission for requesting cluster access. Based on the reason and/or duration of the requested access, access may be restricted in terms of which machines in the cluster can be accessed and how many compute hours can be spent. 

Submitting Jobs 
As of November 26th 2018, submission of jobs is restricted as follows: no user may have more than two jobs in the queue, either active or pending, at the same time, and no job may take longer than 24 hours. This applies over the whole cluster, not per partition! If you think your job takes more than 24 hours, consider splitting it into multiple steps (by epoch(s), for example). If you have more than 2 jobs active at the same time, we will start canceling your oldest jobs until 2 are left. THIS MEANS THAT YOU RISK THE LOSS OF PROGRESS ON YOUR OLDER JOBS IF YOU SUBMIT TOO MANY JOBS . If your job is running for more than 24 hours, we will cancel it. These measures have been instated to prevent individual users from hogging all the CPUs/GPUs. Even if there are CPUs and/or GPUs free on the cluster, users should still abide by these restrictions. If you really need more jobs/running time, and cannot sub-divide your jobs to fit the 24-hour limit, contact the administrators. Based on your situation, we may be able to make a temporary exception. 

The cluster makes use of the SLURM job scheduling system. Everything you want to run on the cluster should be submitted through SLURM. To submit a job, first upload your script to your home folder (On Windows, you can use WinSCP, for example). The job can then be submitted to the cluster by typing 
```
sbatch -p <partition> script.py 
```
where `<partition>` is chromium, calcium, or zirconium (or all if you don't care what partition it runs on). If you want to use a non-standard Python library, please check the section on this page titled “Installing and using extra libraries”. 

By default, jobs do not have access to the cluster GPUs. If you want to make one or more of them available for use, add the `--gres` flag, specifying the NUMBER of GPUs you want to reserve. While your job is running, these GPUs are not available to any other user. An example of reserving one GPU for a job: 
```
sbatch -p <partition> --gres=gpu:1 script.py 
```
Just to repeat: the `gres` flags allows you to select the number of GPUs to reserve, not to select specific GPUs by ID. In the vast majority of cases, your job will need at most one GPU, so please limit your reservations so others can also make use of the GPUs. 

Please isolate your GPU-based code as much as possible. For example, if you need to pre-process data on CPU and then train a model on GPU, please split this into two separate jobs, one without a GPU and one with a GPU. This helps keeping GPUs available for everyone. 

Using Local Modules 
You may run into issues if you try to import local modules (importing local files) since the job scheduling system copies the submitted script to a location on the selected partition. This can be fixed by adding the following lines after the hashbang and before any other imports in your script: 
```
import os 
import sys 
sys.path.append(os.getcwd())  
```

Administration and Monitoring 
A basic monitoring page can be found here. You can use it to see how many GPUs are being used, and to find more information about the jobs you have submitted. 

Some other useful commands: 
- `sinfo`: Shows the list of partitions and their status (if one has status drain, it is not available for jobs). 
- `squeue -u <username>`: Shows the list of pending and running tasks you submitted (PD for pending, R for running). For a complete list of job status codes, see this page. 
- `scancel <job id>`: Cancels a pending or running job. SLURM gives you the job ID when you submit a job, and you can also see the IDs of running jobs with squeue. You can cancel multiple jobs by providing them in a comma-separated fashion. 

Obtaining Results 
When you submit jobs with sbatch, your terminal will not show you the output of your script, it only submits it to the scheduling system. Each job generates an output file named `slurm-<jobid>.out`. This file will be generated wherever the script file is located. 
Output will be written to this file while the job is running, so you can get the latest output by checking the output file with `tail`, for example. Note that the contents of the output file are not “live” i.e. output is buffered and only written to the file at intervals. To force the job to write all output so far to the file immediately, add the line `sys.stdout.flush()` to your script at the point you want it to do so. 

Scripts with Command-Line Arguments 
You may want to submit a job with a script that has one or more command-line arguments, like `myscript.py -v` or `myscript.py --log_dir=~/logs` (you can use Python's built-in argparse module to configure such options, by the way). You shouldn't directly pass these arguments into sbatch together with the script file, because sbatch may interpret them as its own arguments. To correctly pass script arguments, use the `--wrap` option. For example: 
```
sbatch -p chromium --wrap="python3 myscript.py -v" 
```
Note that when using `--wrap` you do not pass in the script file at the end. Also note that you need to provide the interpreter in the `--wrap` statement (python3 in the example above). This is because sbatch will wrap whatever you enter under `--wrap` into a shell command. 

Installing and Using Extra Libraries 
You may need particular Python libaries/modules/frameworks for your project that are not available on the cluster by default. The solution is to create a local Python environment in which you can install them. After logging into datamininghpc, execute the following commands: 
```
> wget https://repo.continuum.io/miniconda/Miniconda3-lates￼ 
> chmod +x Miniconda3-latest-Linux-x86_64.sh 
> ./Miniconda3-latest-Linux-x86_64.sh  
```
At this point, follow the instructions, make sure that you answer yes when asked if the Conda installation should be appended to your bashrc file, and then log out and back in. Then, run the following: 
```
> conda create -n env_name 
> source activate env_name 
> conda install <package> 
```

To submit a job to the job scheduler that uses this environment, you will have to create a separate job file. It can have any name, and does not need a file extension, so for example a file named myscript_job. The contents of this file should look something like this (replace <username> with your username): 
```
#!/bin/bash 
source /home/TUE/<username>/miniconda3/etc/profile.d/conda.sh 
source activate env_name 
python myscript.py 
source deactivate  
```
This tells the scheduler to load Conda, then load the custom environment you have created, run your script, and finally deactivate the environment. The job script can be submitted in the same manner as your other script files, for example: 
```
sbatch -p chromium myscript_job  
```

Interactive Sessions 
The cluster currently does not support interactive sessions. This includes environments such as Jupyter Notebooks. To run such scripts on the cluster, please re-write them into one or more standard Python scripts. 