# Fusion Plasma Simulation

## Todo
- [x] Combine all parquets into one data set so  can normalize the values
  - [ ] Or perhaps just use batch norm?
- [ ] Implement data loader that normalizes the values
- [ ] Implement model and trainer
- [ ] Implement train evaluate loop


# Data

For the labels, 0 = no label, 1 = low confinement mode, 2 = dithering ('inbetween'), 3 = high confinement mode

This is data for full shots (shot = pulse = experiment). We do not want to model the full shot, but rather an area just around a transition in the LHD_label, so zooming in on say ~100ms before/after a transition in the LHD_label


# Running on Cluster
1. Set up ssh keys for passwordless login, use of github deploy keys is recommended.
2. Run  
   ```git clone  git@github.com:Trezorro/fusion-plasma-simulation.git```
3. Upload the data directory with rsync
    ```rsync -avz data/ TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/ ```
    *NOTE: You need the full canonicalized path to the data directory on the server, starting from the home directory.*