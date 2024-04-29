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