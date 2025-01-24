# Snellis access

Login with
```bash
ssh scur2932@snellius.surf.nl
```
Then provide your password.

# Snellius file transfer

# Installing packages
This is not trivial because we need to use the module system. 
This inherently means that the environment is not the same as on your local machine, and that we need to be lucky that the versions match enough to be compatible.

We also cannot use pipenv, because it cannot lock the packages with constraints that are compatible with the module system.

So, we need to have a base list of packages. This can be based on the local Pipfile. But we need to install one by one, and check if it works. 

On snellius, create and activate a virtual environment. 
```bash
source ~/fusion/bin/activate
```

# Generate Keys manually (Linux, Mac, Windows + WSL)

## Generate key pair

First, open a terminal window on your local machine and issue the following command:
```bash
ssh-keygen -t ed25519
```
This will generate a key-pair consisting of a private and a public key. You will be asked in which file you want to store the file, just press enter to use the default (that is, the private key will be stored in `$HOME/.ssh/id_ed25519` and the public key in `$HOME/.ssh/id_ed25519.pub`). Then, you will be asked to choose a passphrase to keep the private key safe (the public key can safely be shared - in fact, you're meant to share it).

## Copy the public key to SURF portal

To get the public key, type on your local machine:
```bash
cat ~/.ssh/id_ed25519.pub
```
The key starts with `ssh-rsa`, then contains a long string of random characters, and ends with some identification of where it was created. Copy the whole key, i.e., starting with `ssh-rsa` and including the identification of where it was created. Now, login to the SURF portal, go to 'Public ssh keys' and click 'add key'. Paste your public key in the SSH key field, and enter the password corresponding to your login in the CUA password field.

## Add private key to local SSH agent

With the key in the SURF portal, the HPC system will authenticate you using the key-pair, rather than your password. However, now, your local PC will ask you for the passphrase every time you want to use the private key to authenticate. That doesn't help much. To have password-less login, you have two options: either you don't enter a passphrase when generating the key-pair (not the safest solution) or you use the local SSH agent to store your passphrase for the duration of the login session on your workstation.

To add the key to the local SSH agent, open a new terminal on your local machine and type:
```bash
ssh-add ~/.ssh/id_ed25519
```
If you get an error “Could not open a connection to your authentication agent”, you may need to start the ssh-agent first (using `eval $(ssh-agent -s)` for a bash shell or `eval $(ssh-agent -c)` for a c-shell).

Finally, to make your life even easier, you can add the key to the SSH agent automatically, by adding:
```plaintext
Host *
    AddKeysToAgent yes
```
to the `$HOME/.ssh/config` file on your local machine.

Now, your local PC should only ask you for the passphrase once per session (i.e., until you log out of your local machine).

# Creating the environment

We need to use the modules system to load the correct versions of the packages. 
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a # vs my installed version 2.2.2
module load plotly.py/5.16.0-GCCcore-12.3.0 # vs 5.24.1
module load SciPy-bundle/2023.07-gfbf-2023a 
```
Then verify that the correct versions are loaded with `python --version` and `pip list`.

Open the following for more information on the [module system](https://servicedesk.surf.nl/wiki/display/WIKI/Environment+Modules).
Most used module commands:

    - `module avail`
        -lists the modules that are available
    - `module load modulename`
        -sets the environment up to use modulename
    - `module add modulename`
        -same as previous
    - `module display modulename`
        -shows what is done when you issue the command `module add modulename`
    - `module unload modulename`
        -removes the environment neccessary for modulename
    - `module rm modulename`
        -same as previous
    - `module list`
        -shows the loaded modules
    - `module help`
        - tells how to use the module command
    - `module spider`
        -The module spider command reports all the modules that can be loaded on a system
    - `module purge`
        -Clear the currently set of loaded modules

# NOTE: maybe venv is not necessary, see [python installation on snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Python#Python-Scipy)
Then create a virtual environment with
```bash
python -m venv fusion
```
Activate the virtual environment with
```bash
source ~/fusion/bin/activate
```
Then install the packages that are not available in the module system via pip.
```bash
pip install --upgrade pip
pip install pyarrow torchinfo omegaconf lightning wandb pytest
```

# Running code on Snellius
We need to pull the code from the repository. 
```bash
git clone <repository>
git fetch
git pull
```
Then we need to run the code. 

Make sure:
- The output folder for the sbatch script exists for the `#SBATCH --output=output/slurms/agent-%j.out` line. 
- 


```bash
